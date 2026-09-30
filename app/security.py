"""Security requirement planning for one protected operation.

Implements ``POST /security/plan``: given an OpenAPI 3.x document, a
``METHOD /path`` operation selector and the security schemes the client
already has available (scheme names plus granted OAuth scopes -- never any
secret values), decide how the operation can be authenticated for an
integration test.

The OpenAPI ``security`` semantics are honored exactly:

* an operation-level ``security`` array **overrides** the root-level one;
  when the operation does not declare ``security`` the root-level array is
  inherited;
* the array items are alternatives (**OR**); the schemes inside one item
  must all be supplied (**AND**);
* an empty array ``security: []`` means the operation is anonymous;
* an item without any scheme (``{}``) is the anonymous alternative;
* absent root and operation security means the operation is anonymous.

Security Scheme Objects referenced from ``components/securitySchemes`` are
resolved through local ``$ref`` chains cycle-safely (OpenAPI 3.1 sibling
fields are merged, 3.0 siblings ignored) and validated for the four
supported types ``apiKey``, ``http``, ``oauth2`` and ``openIdConnect``,
including their required members and oauth2 scope declarations.

The planner picks the satisfiable alternative requiring the fewest
schemes; ties are broken by canonical (sorted-key) JSON encoding. When no
alternative is satisfiable the service still answers ``200`` and lists, per
alternative, the missing schemes or scopes. Dangling / external / circular
references and illegal definitions reject the request with ``422``; no
error or response field ever contains a secret -- only scheme names, scope
names and structural locations are exchanged.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from .refresolver import (
    MAX_REF_CHAIN,
    RefResolver,
    UnresolvableReferenceError,
    is_local_ref,
    parse_local_ref,
)
from .slicer import HTTP_METHODS, parse_selector

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SecurityPlanError(Exception):
    """Base class for security planning failures."""


class InvalidDocumentError(SecurityPlanError):
    """The submitted document is not a recognizable OpenAPI 3.x doc (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidSelectorError(SecurityPlanError):
    """The selector is malformed or unknown (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidSecurityRequestError(SecurityPlanError):
    """The request envelope (``available`` / secret material) is bad (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class SecurityPlanningError(SecurityPlanError):
    """One or more semantic security definition problems (HTTP 422)."""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(
            "; ".join(f"{e['code']} @ {e.get('pointer', '')}" for e in errors)
        )


# --------------------------------------------------------------------------- #
# JSON Pointer helpers (RFC 6901)
# --------------------------------------------------------------------------- #


def escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_join(base: str, token: str) -> str:
    return base + "/" + escape_token(token)


# --------------------------------------------------------------------------- #
# Cycle-safe local $ref resolution (OpenAPI 3.1 sibling merging)
# --------------------------------------------------------------------------- #


def _follow_ref_chain(
    doc: Dict[str, Any],
    node: Any,
    occurrence_pointer: str,
    *,
    merge_siblings: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    """Follow a local ``$ref`` chain starting at *node*.

    Returns ``(resolved_node, terminal_pointer, error)``; exactly one of
    *resolved_node* / *error* is set. With *merge_siblings* enabled
    (OpenAPI 3.1) sibling fields of every layer are overlaid from the
    terminal target back to the referencing occurrence (nearer layers
    win); otherwise the terminal target is used verbatim.
    """
    current = node
    current_pointer = occurrence_pointer
    seen: Dict[str, int] = {}
    refs: List[str] = []
    layers: List[Tuple[Dict[str, Any], str]] = []

    while isinstance(current, dict) and isinstance(current.get("$ref"), str):
        ref = current["$ref"]
        if not is_local_ref(ref):
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": (
                    f"unsupported non-local reference '{ref}': only document-"
                    "local references of the form '#/...' are supported"
                ),
            }
        target_pointer = parse_local_ref(ref)
        if target_pointer in seen:
            loop_start = seen[target_pointer]
            loop = refs[loop_start:] + [ref]
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": "circular reference detected: " + " -> ".join(loop),
            }
        if len(refs) >= MAX_REF_CHAIN:
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": (
                    f"reference chain exceeds the maximum depth of "
                    f"{MAX_REF_CHAIN} links (last reference: {ref})"
                ),
            }
        seen[target_pointer] = len(refs)
        refs.append(ref)
        try:
            target = RefResolver(doc).pointer(target_pointer)
        except UnresolvableReferenceError as exc:
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": exc.ref,
                "error": str(exc),
            }
        if not isinstance(target, dict):
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": f"reference '{ref}' does not target an object",
            }
        layers.append((current, current_pointer))
        current = target
        current_pointer = target_pointer

    if not isinstance(current, dict):
        if refs:
            return None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": refs[-1],
                "error": "reference target must be an object",
            }
        # No $ref was followed: the occurrence itself is not an object; the
        # caller reports it as an invalid definition with its own message.
        return current, current_pointer, None

    layers.append((current, current_pointer))
    if merge_siblings and refs:
        merged: Dict[str, Any] = {}
        for layer_node, _ in reversed(layers):
            for key, value in layer_node.items():
                if key != "$ref":
                    merged[key] = copy.deepcopy(value)
        resolved = merged
    else:
        resolved = copy.deepcopy(current)
    return resolved, current_pointer, None


# --------------------------------------------------------------------------- #
# Shape / envelope validation
# --------------------------------------------------------------------------- #

SUPPORTED_TYPES = frozenset({"apiKey", "http", "oauth2", "openIdConnect"})
APIKEY_LOCATIONS = frozenset({"query", "header", "cookie"})
OAUTH_FLOWS = frozenset(
    {"implicit", "password", "clientCredentials", "authorizationCode"}
)
# Flow name -> OAuth Flow members that must be non-empty URL strings.
FLOW_REQUIRED_URLS = {
    "implicit": ("authorizationUrl",),
    "password": ("tokenUrl",),
    "clientCredentials": ("tokenUrl",),
    "authorizationCode": ("authorizationUrl", "tokenUrl"),
}

# Top-level request keys that would carry secret material. The endpoint
# contract only exchanges scheme and scope *names*; the presence of any of
# these keys rejects the request (the value is never read or echoed).
FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "secret",
        "secrets",
        "credential",
        "credentials",
        "token",
        "tokens",
        "password",
        "passwords",
        "api_key",
        "apikey",
        "client_secret",
        "clientsecret",
    }
)


def _validate_shape(doc: Any) -> List[str]:
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    messages: List[str] = []
    version = doc.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        messages.append(
            "document.openapi must be a version string starting with '3.' "
            f"(OpenAPI 3.x), got {version!r}"
        )
    if not isinstance(doc.get("paths"), dict):
        messages.append("document.paths must be an object")
    return messages


def _declared_scheme_names(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    components = doc.get("components")
    if not isinstance(components, dict):
        return None
    schemes = components.get("securitySchemes")
    if not isinstance(schemes, dict):
        return None
    return schemes


def _validate_available(
    available: Any, doc: Dict[str, Any]
) -> Tuple[Optional[Dict[str, List[str]]], List[str]]:
    """Validate the ``available`` envelope into a name->granted-scope map."""
    if not isinstance(available, dict):
        return None, ["available must be an object mapping scheme names to "
                      "arrays of granted scope name strings"]
    declared = _declared_scheme_names(doc)
    normalized: Dict[str, List[str]] = {}
    messages: List[str] = []
    for name in sorted(available):
        if not isinstance(name, str) or not name:
            messages.append("available keys must be non-empty scheme name strings")
            continue
        scopes = available[name]
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            messages.append(
                f"available.{name} must be an array of granted scope name strings"
            )
            continue
        if declared is None or name not in declared:
            messages.append(
                f"available.{name}: no security scheme with that name is "
                "declared in components/securitySchemes"
            )
            continue
        normalized[name] = list(scopes)
    return normalized, messages


# --------------------------------------------------------------------------- #
# Security Scheme Object validation
# --------------------------------------------------------------------------- #


def _scheme_error(pointer: str, name: str, message: str) -> Dict[str, Any]:
    return {
        "code": "invalid_security_scheme",
        "pointer": pointer,
        "scheme": name,
        "error": message,
    }


def _validate_oauth2(
    scheme_name: str, node: Dict[str, Any], pointer: str
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    flows = node.get("flows")
    if not isinstance(flows, dict):
        errors.append(
            _scheme_error(
                pointer,
                scheme_name,
                "oauth2 security scheme must declare an object 'flows'",
            )
        )
        return errors
    known = [name for name in flows if name in OAUTH_FLOWS]
    if not known:
        errors.append(
            _scheme_error(
                pointer,
                scheme_name,
                "oauth2 'flows' must define at least one of implicit, "
                "password, clientCredentials or authorizationCode",
            )
        )
        return errors
    for flow_name in known:
        flow = flows[flow_name]
        flow_pointer = pointer_join(pointer_join(pointer, "flows"), flow_name)
        if not isinstance(flow, dict):
            errors.append(
                _scheme_error(
                    flow_pointer,
                    scheme_name,
                    f"oauth2 flow '{flow_name}' must be an object",
                )
            )
            continue
        for member in FLOW_REQUIRED_URLS[flow_name]:
            value = flow.get(member)
            if not isinstance(value, str) or not value:
                errors.append(
                    _scheme_error(
                        pointer_join(flow_pointer, member),
                        scheme_name,
                        f"oauth2 flow '{flow_name}' requires a non-empty "
                        f"string '{member}'",
                    )
                )
        scopes = flow.get("scopes")
        if not isinstance(scopes, dict):
            errors.append(
                _scheme_error(
                    pointer_join(flow_pointer, "scopes"),
                    scheme_name,
                    f"oauth2 flow '{flow_name}' requires an object 'scopes' "
                    "mapping scope name to description",
                )
            )
        elif not all(isinstance(v, str) for v in scopes.values()):
            errors.append(
                _scheme_error(
                    pointer_join(flow_pointer, "scopes"),
                    scheme_name,
                    f"oauth2 flow '{flow_name}' scope descriptions must be strings",
                )
            )
        refresh_url = flow.get("refreshUrl")
        if refresh_url is not None and not isinstance(refresh_url, str):
            errors.append(
                _scheme_error(
                    pointer_join(flow_pointer, "refreshUrl"),
                    scheme_name,
                    f"oauth2 flow '{flow_name}' 'refreshUrl' must be a string",
                )
            )
    return errors


def _validate_scheme_definition(
    scheme_name: str, node: Dict[str, Any], pointer: str
) -> List[Dict[str, Any]]:
    """Validate one resolved Security Scheme Object definition."""
    errors: List[Dict[str, Any]] = []
    scheme_type = node.get("type")
    if not isinstance(scheme_type, str) or not scheme_type:
        errors.append(
            _scheme_error(
                pointer,
                scheme_name,
                "security scheme must declare a non-empty string 'type' of "
                "apiKey, http, oauth2 or openIdConnect",
            )
        )
        return errors
    if scheme_type not in SUPPORTED_TYPES:
        errors.append(
            _scheme_error(
                pointer_join(pointer, "type"),
                scheme_name,
                f"unsupported security scheme type {scheme_type!r}: expected "
                "one of apiKey, http, oauth2, openIdConnect",
            )
        )
        return errors

    if scheme_type == "apiKey":
        location = node.get("in")
        if not isinstance(location, str) or location not in APIKEY_LOCATIONS:
            errors.append(
                _scheme_error(
                    pointer_join(pointer, "in"),
                    scheme_name,
                    "apiKey security scheme requires 'in' to be one of "
                    "query, header, cookie",
                )
            )
        key_name = node.get("name")
        if not isinstance(key_name, str) or not key_name:
            errors.append(
                _scheme_error(
                    pointer_join(pointer, "name"),
                    scheme_name,
                    "apiKey security scheme requires a non-empty string 'name'",
                )
            )
    elif scheme_type == "http":
        http_scheme = node.get("scheme")
        if not isinstance(http_scheme, str) or not http_scheme:
            errors.append(
                _scheme_error(
                    pointer_join(pointer, "scheme"),
                    scheme_name,
                    "http security scheme requires a non-empty string "
                    "'scheme' (e.g. 'basic' or 'bearer')",
                )
            )
        bearer_format = node.get("bearerFormat")
        if bearer_format is not None and not isinstance(bearer_format, str):
            errors.append(
                _scheme_error(
                    pointer_join(pointer, "bearerFormat"),
                    scheme_name,
                    "http security scheme 'bearerFormat' must be a string",
                )
            )
    elif scheme_type == "openIdConnect":
        url = node.get("openIdConnectUrl")
        if not isinstance(url, str) or not url:
            errors.append(
                _scheme_error(
                    pointer_join(pointer, "openIdConnectUrl"),
                    scheme_name,
                    "openIdConnect security scheme requires a non-empty "
                    "string 'openIdConnectUrl'",
                )
            )
    else:  # oauth2
        errors.extend(_validate_oauth2(scheme_name, node, pointer))
    return errors


def _oauth_declared_scopes(node: Dict[str, Any]) -> List[str]:
    """Union of scope names declared by every oauth2 flow (valid node)."""
    scopes: List[str] = []
    flows = node.get("flows")
    if isinstance(flows, dict):
        for flow_name in FLOW_REQUIRED_URLS:
            flow = flows.get(flow_name)
            if isinstance(flow, dict) and isinstance(flow.get("scopes"), dict):
                scopes.extend(flow["scopes"])
    return scopes


# --------------------------------------------------------------------------- #
# Injection descriptors
# --------------------------------------------------------------------------- #


def _build_inject(name: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """Describe where the chosen credential has to be injected at runtime."""
    scheme_type = node["type"]
    if scheme_type == "apiKey":
        location = node["in"]
        key_name = node["name"]
        if location == "header":
            description = (
                f"Inject the API key for scheme '{name}' into the "
                f"'{key_name}' request header as the raw value (no prefix)."
            )
        elif location == "query":
            description = (
                f"Inject the API key for scheme '{name}' into the "
                f"'{key_name}' query parameter of the request target."
            )
        else:
            description = (
                f"Inject the API key for scheme '{name}' into the "
                f"'{key_name}' cookie, sent in the Cookie header."
            )
        return {
            "location": location,
            "name": key_name,
            "http_scheme": None,
            "description": description,
        }

    if scheme_type == "http":
        http_scheme = node["scheme"]
        inject: Dict[str, Any] = {
            "location": "authorization",
            "name": "Authorization",
            "http_scheme": http_scheme,
        }
        lowered = http_scheme.lower()
        if lowered == "basic":
            inject["description"] = (
                f"Inject HTTP Basic credentials for scheme '{name}' into the "
                "'Authorization' header as 'Basic <base64(user:password)>'."
            )
        elif lowered == "bearer":
            detail = ""
            bearer_format = node.get("bearerFormat")
            if isinstance(bearer_format, str) and bearer_format:
                detail = f" (bearer format: {bearer_format})"
            inject["description"] = (
                f"Inject the bearer token for scheme '{name}' into the "
                "'Authorization' header as 'Bearer <token>'" + detail + "."
            )
        else:
            inject["description"] = (
                f"Inject HTTP {http_scheme} credentials for scheme '{name}' "
                "into the 'Authorization' header using the "
                f"'{http_scheme.capitalize()}' authentication scheme."
            )
        return inject

    if scheme_type == "oauth2":
        return {
            "location": "authorization",
            "name": "Authorization",
            "http_scheme": "bearer",
            "description": (
                f"Inject the OAuth 2 access token for scheme '{name}' into "
                "the 'Authorization' header as 'Bearer <token>'; the token "
                "must carry the required scopes."
            ),
        }

    # openIdConnect
    return {
        "location": "authorization",
        "name": "Authorization",
        "http_scheme": "bearer",
        "description": (
            f"Inject the OpenID Connect token for scheme '{name}' into the "
            "'Authorization' header as 'Bearer <token>'."
        ),
    }


# --------------------------------------------------------------------------- #
# Effective security requirement evaluation
# --------------------------------------------------------------------------- #


class _ResolvedScheme:
    """A scheme occurrence after cycle-safe $ref resolution."""

    __slots__ = ("name", "node", "pointer", "scopes")

    def __init__(
        self,
        name: str,
        node: Dict[str, Any],
        pointer: str,
        scopes: List[str],
    ) -> None:
        self.name = name
        self.node = node
        # JSON Pointer of the terminal Security Scheme Object definition.
        self.pointer = pointer
        self.scopes = scopes


def _evaluate_security(
    doc: Dict[str, Any],
    security: Any,
    base_pointer: str,
    is31: bool,
) -> Tuple[List[List[Optional[_ResolvedScheme]]], List[Dict[str, Any]]]:
    """Validate the effective security array into resolved alternatives.

    Returns ``(alternatives, errors)``; each alternative lines up with one
    requirement array item and holds the AND-group of resolved schemes
    (``None`` slots mark occurrences that already produced an error -- the
    whole request is rejected on *any* error, so they never reach
    satisfiability analysis).
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(security, list):
        errors.append(
            {
                "code": "invalid_security",
                "pointer": base_pointer,
                "error": "'security' must be an array of requirement objects",
            }
        )
        return [], errors

    declared = _declared_scheme_names(doc)
    # A scheme may be referenced by several alternatives; resolve and
    # validate its definition only once, keyed by scheme name.
    resolved_schemes: Dict[str, Optional[_ResolvedScheme]] = {}

    def resolve_named(scheme_name: str) -> Optional[_ResolvedScheme]:
        if scheme_name in resolved_schemes:
            return resolved_schemes[scheme_name]
        if declared is None or scheme_name not in declared:
            resolved_schemes[scheme_name] = None
            return None

        definition_pointer = pointer_join(
            "/components/securitySchemes", scheme_name
        )
        resolved, terminal_pointer, ref_error = _follow_ref_chain(
            doc,
            declared[scheme_name],
            definition_pointer,
            merge_siblings=is31,
        )
        if ref_error is not None:
            ref_error["scheme"] = scheme_name
            errors.append(ref_error)
            resolved_schemes[scheme_name] = None
            return None
        if not isinstance(resolved, dict):
            errors.append(
                {
                    "code": "invalid_security_scheme",
                    "pointer": terminal_pointer,
                    "scheme": scheme_name,
                    "error": "security scheme definition must be an object",
                }
            )
            resolved_schemes[scheme_name] = None
            return None

        errors.extend(
            _validate_scheme_definition(scheme_name, resolved, terminal_pointer)
        )
        entry = _ResolvedScheme(scheme_name, resolved, terminal_pointer, [])
        resolved_schemes[scheme_name] = entry
        return entry

    alternatives: List[List[Optional[_ResolvedScheme]]] = []

    for item_index, item in enumerate(security):
        item_pointer = pointer_join(base_pointer, str(item_index))
        if not isinstance(item, dict):
            errors.append(
                {
                    "code": "invalid_security",
                    "pointer": item_pointer,
                    "error": (
                        "each security requirement must be an object mapping "
                        "security scheme names to required scope arrays"
                    ),
                }
            )
            alternatives.append([])
            continue

        group: List[Optional[_ResolvedScheme]] = []
        for scheme_name in item:
            requirement_pointer = pointer_join(item_pointer, scheme_name)
            raw_scopes = item[scheme_name]
            if not isinstance(raw_scopes, list) or not all(
                isinstance(scope, str) for scope in raw_scopes
            ):
                errors.append(
                    {
                        "code": "invalid_security",
                        "pointer": requirement_pointer,
                        "scheme": scheme_name,
                        "error": (
                            "the scope list of a security requirement must be "
                            "an array of scope name strings"
                        ),
                    }
                )
                continue

            resolved = resolve_named(scheme_name)
            if resolved is None:
                # resolve_named records reference/definition errors, but an
                # undeclared name is an occurrence-level error: emit it once
                # per requirement occurrence at that occurrence's pointer.
                if declared is None or scheme_name not in declared:
                    errors.append(
                        {
                            "code": "unknown_scheme",
                            "pointer": requirement_pointer,
                            "scheme": scheme_name,
                            "error": (
                                f"security requirement references scheme "
                                f"'{scheme_name}', which is not declared in "
                                "components/securitySchemes"
                            ),
                        }
                    )
                continue

            occurrence = _ResolvedScheme(
                scheme_name,
                resolved.node,
                resolved.pointer,
                list(raw_scopes),
            )

            # Scope usage rules for this occurrence.
            scheme_type = occurrence.node.get("type")
            if scheme_type in ("apiKey", "http") and occurrence.scopes:
                errors.append(
                    {
                        "code": "unexpected_scope",
                        "pointer": requirement_pointer,
                        "scheme": scheme_name,
                        "error": (
                            f"{scheme_type} security scheme '{scheme_name}' "
                            "must not require scopes; only oauth2 and "
                            "openIdConnect requirements carry scope lists"
                        ),
                    }
                )
            elif scheme_type == "oauth2":
                declared_scopes = set(
                    _oauth_declared_scopes(occurrence.node)
                )
                for scope in occurrence.scopes:
                    if scope not in declared_scopes:
                        errors.append(
                            {
                                "code": "undeclared_scope",
                                "pointer": requirement_pointer,
                                "scheme_pointer": occurrence.pointer,
                                "scheme": scheme_name,
                                "scope": scope,
                                "error": (
                                    f"required scope '{scope}' is not declared "
                                    f"by any flow of oauth2 scheme "
                                    f"'{scheme_name}'"
                                ),
                            }
                        )
            group.append(occurrence)
        alternatives.append(group)
    return alternatives, errors


def _sorted_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]):
        return (
            str(item.get("code")),
            str(item.get("pointer") or ""),
            str(item.get("scheme") or ""),
            str(item.get("scope") or ""),
            str(item.get("ref") or ""),
            str(item.get("error") or ""),
        )

    return sorted(errors, key=key)


def _canonical_key(item: Any) -> str:
    """Canonical JSON encoding of a security requirement object."""
    return json.dumps(
        item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def plan_security(
    payload: Any,
) -> Dict[str, Any]:
    """Plan the security injection for one operation.

    :raises InvalidDocumentError: *document* is not OpenAPI 3.x (400).
    :raises InvalidSelectorError: the selector is malformed/unknown (400).
    :raises InvalidSecurityRequestError: bad ``available`` envelope or
        secret material in the request (400).
    :raises SecurityPlanningError: malformed security definitions, unknown
        schemes, dangling/external/circular references or illegal scope
        usage (422).
    """
    if not isinstance(payload, dict):
        raise InvalidSecurityRequestError(["request body must be a JSON object"])

    forbidden = sorted(
        key
        for key in payload
        if isinstance(key, str) and key.lower() in FORBIDDEN_REQUEST_KEYS
    )
    if forbidden:
        raise InvalidSecurityRequestError(
            [
                "the request must not carry secret material (detected an "
                f"'{forbidden[0]}' member): send only the OpenAPI document, "
                "the operation selector, available scheme names and granted "
                "scope names; credential values are never accepted"
            ]
        )

    doc = payload.get("document")
    shape_errors = _validate_shape(doc)
    if shape_errors:
        raise InvalidDocumentError(shape_errors)

    parsed = parse_selector(payload.get("selector"))
    if parsed is None:
        raise InvalidSelectorError(
            [
                f"malformed selector {payload.get('selector')!r}: expected "
                "'METHOD /path' (e.g. 'GET /pets/{petId}')"
            ]
        )
    method, path = parsed
    if path not in doc["paths"]:
        raise InvalidSelectorError([f"{method.upper()} {path}: path is not declared"])

    available, envelope_errors = _validate_available(
        payload.get("available", {}), doc
    )
    if envelope_errors:
        raise InvalidSecurityRequestError(envelope_errors)

    is31 = str(doc.get("openapi", "")).startswith(("3.1", "3.2"))
    path_pointer = pointer_join("/paths", path)

    path_item, _terminal_pointer, item_error = _follow_ref_chain(
        doc, doc["paths"][path], path_pointer, merge_siblings=is31
    )
    if item_error is not None:
        raise SecurityPlanningError(_sorted_errors([item_error]))
    if not isinstance(path_item, dict):
        raise SecurityPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_security",
                        "pointer": _terminal_pointer,
                        "error": "path item must be an object",
                    }
                ]
            )
        )

    method_keys = {
        key.lower(): key for key in path_item if key.lower() in HTTP_METHODS
    }
    if method not in method_keys:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: operation is not declared"]
        )
    operation_key = method_keys[method]
    operation_pointer = pointer_join(path_pointer, operation_key)
    operation = path_item[operation_key]
    if not isinstance(operation, dict):
        raise SecurityPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_security",
                        "pointer": operation_pointer,
                        "error": "operation must be an object",
                    }
                ]
            )
        )

    # Operation-level security overrides root-level security; omission
    # inherits the root array; an explicit [] (or no security anywhere)
    # makes the operation anonymous.
    if "security" in operation:
        security = operation["security"]
        base_pointer = pointer_join(operation_pointer, "security")
        security_source = "operation"
    elif "security" in doc:
        security = doc["security"]
        base_pointer = "/security"
        security_source = "root"
    else:
        security = None
        base_pointer = ""
        security_source = "none"

    result: Dict[str, Any] = {
        "selector": f"{method.upper()} {path}",
        "security_source": security_source,
    }

    if security is None:
        result.update(
            {
                "anonymous": True,
                "satisfiable": True,
                "selected": None,
                "alternatives": [],
            }
        )
        return result

    alternatives, errors = _evaluate_security(
        doc, security, base_pointer, is31
    )
    if errors:
        raise SecurityPlanningError(_sorted_errors(errors))

    # Anonymous empty array: the OR of zero alternatives allows every
    # request, unconditionally. (A zero-scheme item '{}' inside a non-empty
    # array is instead an *optional* anonymous alternative and flows
    # through regular selection below.)
    if security == []:
        result.update(
            {
                "anonymous": True,
                "satisfiable": True,
                "selected": None,
                "alternatives": [],
            }
        )
        return result

    # ------------------------------------------------------------------ #
    # Satisfiability against 'available'
    # ------------------------------------------------------------------ #
    alternative_reports: List[Dict[str, Any]] = []
    satisfiable_indices: List[int] = []
    for index, group in enumerate(alternatives):
        item_pointer = pointer_join(base_pointer, str(index))
        requirement_item = security[index]
        missing: List[Dict[str, Any]] = []
        for scheme in group:
            granted = available.get(scheme.name)
            if granted is None:
                missing.append(
                    {
                        "kind": "scheme",
                        "scheme": scheme.name,
                        "reason": (
                            f"security scheme '{scheme.name}' is not listed "
                            "as available"
                        ),
                    }
                )
                continue
            if scheme.node.get("type") in ("oauth2", "openIdConnect"):
                granted_set = set(granted)
                for scope in scheme.scopes:
                    if scope not in granted_set:
                        missing.append(
                            {
                                "kind": "scope",
                                "scheme": scheme.name,
                                "scope": scope,
                                "reason": (
                                    f"required scope '{scope}' of "
                                    f"{scheme.node['type']} scheme "
                                    f"'{scheme.name}' has not been granted"
                                ),
                            }
                        )
        satisfied = not missing
        if satisfied:
            satisfiable_indices.append(index)

        def missing_key(entry: Dict[str, Any]):
            return (
                str(entry.get("scheme") or ""),
                str(entry.get("kind") or ""),
                str(entry.get("scope") or ""),
                str(entry.get("reason") or ""),
            )

        alternative_reports.append(
            {
                "item_index": index,
                "item_pointer": item_pointer,
                "satisfiable": satisfied,
                "schemes": sorted(requirement_item),
                "missing": sorted(missing, key=missing_key),
            }
        )

    # Pick the satisfiable alternative with the fewest schemes; ties are
    # broken by canonical (sorted-key, compact) JSON of the requirement.
    chosen_index: Optional[int] = None
    if satisfiable_indices:
        chosen_index = min(
            satisfiable_indices,
            key=lambda i: (
                len(security[i]),
                _canonical_key(security[i]),
                i,
            ),
        )

    selected: Optional[Dict[str, Any]] = None
    if chosen_index is not None:
        group = alternatives[chosen_index]
        chosen_schemes: List[Dict[str, Any]] = []
        for scheme in sorted(group, key=lambda s: s.name):
            granted = available.get(scheme.name, [])
            if scheme.node.get("type") in ("oauth2", "openIdConnect"):
                granted_scopes = sorted(granted)
            else:
                granted_scopes = []
            occurrence = pointer_join(
                pointer_join(base_pointer, str(chosen_index)), scheme.name
            )
            chosen_schemes.append(
                {
                    "name": scheme.name,
                    "type": scheme.node["type"],
                    "required_scopes": sorted(scheme.scopes),
                    "granted_scopes": granted_scopes,
                    "scheme_pointer": scheme.pointer,
                    "requirement_pointer": occurrence,
                    "inject": _build_inject(scheme.name, scheme.node),
                }
            )
        selected = {
            "item_index": chosen_index,
            "item_pointer": pointer_join(base_pointer, str(chosen_index)),
            "schemes": chosen_schemes,
        }

    result.update(
        {
            "anonymous": False,
            "satisfiable": chosen_index is not None,
            "selected": selected,
            "alternatives": alternative_reports,
        }
    )
    return result
