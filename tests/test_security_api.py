"""Regression tests for ``POST /security/plan``.

Covers operation-level override of root security, OR-between-items /
AND-within-an-item semantics, ``security: []`` anonymity, cycle-safe
local Security Scheme ``$ref`` resolution (3.0 terminal vs 3.1 sibling
merging), apiKey/http/oauth2/openIdConnect definition and scope
validation, fewest-scheme selection with canonical-JSON tie-breaking,
injection descriptors with source JSON Pointers, per-alternative missing
reports when nothing is satisfiable (still 200), 422 handling for
dangling/external/circular references and illegal definitions, 400
handling for bad documents/selectors/envelopes, secret rejection without
echoing values, and the non-mutation guarantee.
"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def api_key(name="apiKey", *, location="header", key_name="X-API-Key"):
    return {"type": "apiKey", "in": location, "name": key_name}


def http_scheme(name="basic", *, scheme="basic", bearer_format=None):
    node = {"type": "http", "scheme": scheme}
    if bearer_format is not None:
        node["bearerFormat"] = bearer_format
    return node


def oauth2(name="oauth", *, flows=None):
    if flows is None:
        flows = {
            "authorizationCode": {
                "authorizationUrl": "https://idp.example.com/authorize",
                "tokenUrl": "https://idp.example.com/token",
                "refreshUrl": "https://idp.example.com/refresh",
                "scopes": {
                    "pets:read": "read pets",
                    "pets:write": "write pets",
                    "admin": "administer",
                },
            }
        }
    return {"type": "oauth2", "flows": flows}


def oidc(name="oidc"):
    return {
        "type": "openIdConnect",
        "openIdConnectUrl": "https://idp.example.com/.well-known/openid-configuration",
    }


def make_doc(schemes=None, *, paths=None, openapi="3.0.3", security=None,
             components_extra=None):
    components = {}
    if schemes is not None:
        components["securitySchemes"] = schemes
    if components_extra:
        components.update(components_extra)
    doc = {
        "openapi": openapi,
        "info": {"title": "Pet API", "version": "1.0"},
        "paths": paths if paths is not None else {
            "/pets": {"get": {"responses": {"200": {"description": "ok"}}}}
        },
    }
    if security is not None:
        doc["security"] = security
    if components:
        doc["components"] = components
    return doc


def op(security=..., **extra):
    operation = {"responses": {"200": {"description": "ok"}}}
    operation.update(extra)
    if security is not ...:
        operation["security"] = security
    return operation


def plan(doc, selector="GET /pets", available=None, **extra):
    payload = {"document": doc, "selector": selector}
    if available is not None:
        payload["available"] = available
    payload.update(extra)
    return client.post("/security/plan", json=payload)


# --------------------------------------------------------------------------- #
# Basic inheritance and anonymity
# --------------------------------------------------------------------------- #


def test_no_security_anywhere_is_anonymous():
    doc = make_doc()
    resp = plan(doc)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["anonymous"] is True
    assert body["satisfiable"] is True
    assert body["selected"] is None
    assert body["alternatives"] == []
    assert body["security_source"] == "none"


def test_empty_security_array_is_anonymous_even_with_root_security():
    doc = make_doc(
        schemes={"apiKey": api_key()},
        security=[{"apiKey": []}],
        paths={"/pets": {"get": op(security=[])}},
    )
    resp = plan(doc, available={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["anonymous"] is True
    assert body["security_source"] == "operation"
    assert body["selected"] is None
    assert body["alternatives"] == []


def test_root_security_is_inherited_when_operation_omits_it():
    doc = make_doc(schemes={"basic": http_scheme()}, security=[{"basic": []}])
    resp = plan(doc, available={"basic": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["anonymous"] is False
    assert body["security_source"] == "root"
    selected = body["selected"]
    assert selected["item_pointer"] == "/security/0"
    assert selected["schemes"][0]["name"] == "basic"
    assert selected["schemes"][0]["requirement_pointer"] == "/security/0/basic"


def test_operation_security_overrides_root_security():
    doc = make_doc(
        schemes={"basic": http_scheme(), "apiKey": api_key()},
        security=[{"basic": []}],
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    resp = plan(doc, available={"basic": [], "apiKey": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["security_source"] == "operation"
    names = [s["name"] for s in body["selected"]["schemes"]]
    assert names == ["apiKey"]
    # The root requirement is not among the alternatives.
    assert [a["schemes"] for a in body["alternatives"]] == [["apiKey"]]


def test_empty_operation_security_overrides_root_and_is_anonymous():
    doc = make_doc(
        schemes={"basic": http_scheme()},
        security=[{"basic": []}],
        paths={"/pets": {"get": op(security=[])}},
    )
    resp = plan(doc, available={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["anonymous"] is True


# --------------------------------------------------------------------------- #
# OR / AND semantics and selection
# --------------------------------------------------------------------------- #


def test_items_are_or_picks_satisfiable_one():
    doc = make_doc(
        schemes={"basic": http_scheme(), "apiKey": api_key()},
        paths={"/pets": {"get": op(security=[{"basic": []}, {"apiKey": []}])}},
    )
    # Only the apiKey alternative is available.
    resp = plan(doc, available={"apiKey": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["selected"]["item_index"] == 1
    alt0, alt1 = body["alternatives"]
    assert alt0["satisfiable"] is False
    assert alt0["missing"][0]["kind"] == "scheme"
    assert alt0["missing"][0]["scheme"] == "basic"
    assert alt1["satisfiable"] is True
    assert alt1["missing"] == []


def test_schemes_within_an_item_are_and_all_must_be_available():
    doc = make_doc(
        schemes={"a": api_key(key_name="A"), "b": api_key(key_name="B")},
        paths={"/pets": {"get": op(security=[{"a": [], "b": []}])}},
    )
    resp = plan(doc, available={"a": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["satisfiable"] is False
    assert body["selected"] is None
    missing = body["alternatives"][0]["missing"]
    assert [(m["kind"], m["scheme"]) for m in missing] == [("scheme", "b")]

    resp2 = plan(doc, available={"a": [], "b": []})
    assert resp2.json()["satisfiable"] is True
    names = [s["name"] for s in resp2.json()["selected"]["schemes"]]
    assert names == ["a", "b"]  # selected schemes sorted by name


def test_and_item_reports_both_missing_scheme_and_scope():
    doc = make_doc(
        schemes={
            "oauth": {
                "type": "oauth2",
                "flows": {
                    "implicit": {
                        "authorizationUrl": "https://idp/auth",
                        "scopes": {"a": "A", "b": "B"},
                    }
                },
            },
            "k": api_key(key_name="K"),
        },
        paths={"/pets": {"get": op(
            security=[{"oauth": ["a", "b"], "k": []}]
        )}},
    )
    body = plan(doc, available={"oauth": ["a"]}).json()
    missing = body["alternatives"][0]["missing"]
    # Both the unavailable 'k' credential and the missing 'b' scope are
    # reported, stably sorted by scheme/kind.
    assert [(m["kind"], m["scheme"], m.get("scope")) for m in missing] == [
        ("scheme", "k", None),
        ("scope", "oauth", "b"),
    ]


def test_fewest_scheme_item_is_selected():
    doc = make_doc(
        schemes={
            "a": api_key(key_name="A"),
            "b": api_key(key_name="B"),
            "c": api_key(key_name="C"),
        },
        paths={"/pets": {"get": op(
            security=[{"a": [], "b": []}, {"c": []}]
        )}},
    )
    resp = plan(doc, available={"a": [], "b": [], "c": []})
    body = resp.json()
    # The one-scheme item beats the two-scheme item even though both work.
    assert body["selected"]["item_index"] == 1
    assert [s["name"] for s in body["selected"]["schemes"]] == ["c"]


def test_ties_are_broken_by_canonical_json_sorting():
    # Two single-scheme items with names 'z' and 'a': canonical JSON of
    # {"a":[]} sorts before {"z":[]}, so 'a' wins regardless of order.
    doc = make_doc(
        schemes={"a": api_key(key_name="A"), "z": api_key(key_name="Z")},
        paths={"/pets": {"get": op(security=[{"z": []}, {"a": []}])}},
    )
    resp = plan(doc, available={"a": [], "z": []})
    assert resp.json()["selected"]["item_index"] == 1

    # Same outcome with the item order reversed (index of the winner moves,
    # but it is always the 'a' requirement).
    doc2 = make_doc(
        schemes={"a": api_key(key_name="A"), "z": api_key(key_name="Z")},
        paths={"/pets": {"get": op(security=[{"a": []}, {"z": []}])}},
    )
    resp2 = plan(doc2, available={"a": [], "z": []})
    assert resp2.json()["selected"]["item_index"] == 0
    assert resp2.json()["selected"]["schemes"][0]["name"] == "a"


def test_canonical_tie_break_considers_scopes():
    # Same single scheme, different scope lists: the lexicographically
    # smaller canonical scope set wins.
    doc = make_doc(
        schemes={"oauth": oauth2()},
        paths={"/pets": {"get": op(security=[
            {"oauth": ["pets:write"]},
            {"oauth": ["pets:read"]},
        ])}},
    )
    resp = plan(
        doc,
        available={"oauth": ["pets:read", "pets:write", "admin"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["selected"]["item_index"] == 1
    assert body["selected"]["schemes"][0]["required_scopes"] == ["pets:read"]


def test_empty_scheme_item_is_an_anonymous_alternative():
    # security: [{}] means authentication is *optional*: it is selected as
    # a zero-scheme requirement but the operation is not reported as
    # unconditionally anonymous.
    doc = make_doc(
        schemes={"basic": http_scheme()},
        paths={"/pets": {"get": op(security=[{"basic": []}, {}])}},
    )
    resp = plan(doc, available={})
    body = resp.json()
    assert resp.status_code == 200, resp.text
    assert body["anonymous"] is False
    assert body["satisfiable"] is True
    selected = body["selected"]
    assert selected["item_index"] == 1
    assert selected["schemes"] == []
    assert [a["schemes"] for a in body["alternatives"]] == [["basic"], []]


# --------------------------------------------------------------------------- #
# Scope satisfaction (oauth2 / openIdConnect)
# --------------------------------------------------------------------------- #


def test_oauth2_missing_scope_is_reported():
    doc = make_doc(
        schemes={"oauth": oauth2()},
        paths={"/pets": {"get": op(
            security=[{"oauth": ["pets:read", "pets:write"]}]
        )}},
    )
    resp = plan(doc, available={"oauth": ["pets:read"]})
    body = resp.json()
    assert resp.status_code == 200, resp.text
    assert body["satisfiable"] is False
    assert body["selected"] is None
    missing = body["alternatives"][0]["missing"]
    assert missing == [
        {
            "kind": "scope",
            "scheme": "oauth",
            "scope": "pets:write",
            "reason": (
                "required scope 'pets:write' of oauth2 scheme 'oauth' has "
                "not been granted"
            ),
        }
    ]


def test_oauth2_satisfied_selects_scheme_with_sorted_scopes():
    doc = make_doc(
        schemes={"oauth": oauth2()},
        paths={"/pets": {"get": op(
            security=[{"oauth": ["pets:write", "pets:read"]}]
        )}},
    )
    resp = plan(
        doc,
        available={"oauth": ["pets:write", "pets:read", "admin"]},
    )
    body = resp.json()
    assert body["satisfiable"] is True
    scheme = body["selected"]["schemes"][0]
    assert scheme["required_scopes"] == ["pets:read", "pets:write"]
    assert scheme["granted_scopes"] == ["admin", "pets:read", "pets:write"]


def test_openidconnect_scopes_are_checked_against_grants():
    doc = make_doc(
        schemes={"oidc": oidc()},
        paths={"/pets": {"get": op(security=[{"oidc": ["openid", "email"]}])}},
    )
    missing = plan(doc, available={"oidc": ["openid"]}).json()
    assert missing["satisfiable"] is False
    assert missing["alternatives"][0]["missing"][0]["scope"] == "email"

    ok = plan(doc, available={"oidc": ["email", "openid"]}).json()
    assert ok["satisfiable"] is True
    assert ok["selected"]["schemes"][0]["granted_scopes"] == ["email", "openid"]


def test_apikey_and_http_ignore_available_scope_names():
    doc = make_doc(
        schemes={"basic": http_scheme(), "apiKey": api_key()},
        paths={"/pets": {"get": op(
            security=[{"basic": []}, {"apiKey": []}]
        )}},
    )
    # Scope names granted to a non-OAuth scheme are meaningless but do not
    # block availability of the credential.
    resp = plan(doc, available={"apiKey": ["anything"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["selected"]["schemes"][0]["name"] == "apiKey"
    assert body["selected"]["schemes"][0]["granted_scopes"] == []


# --------------------------------------------------------------------------- #
# Injection descriptors
# --------------------------------------------------------------------------- #


def test_apikey_header_query_cookie_injection():
    for location, key_name, expected_location in (
        ("header", "X-API-Key", "header"),
        ("query", "api_key", "query"),
        ("cookie", "session", "cookie"),
    ):
        scheme_name = f"key_{location}"
        doc = make_doc(
            schemes={scheme_name: api_key(
                location=location, key_name=key_name
            )},
            paths={"/pets": {"get": op(
                security=[{scheme_name: []}]
            )}},
        )
        body = plan(doc, available={scheme_name: []}).json()
        inject = body["selected"]["schemes"][0]["inject"]
        assert inject["location"] == expected_location
        assert inject["name"] == key_name
        assert inject["http_scheme"] is None
        assert key_name in inject["description"]


def test_http_basic_bearer_and_other_scheme_injection():
    cases = {
        "basic": (http_scheme(scheme="basic"), "basic"),
        "bearer": (http_scheme(scheme="bearer", bearer_format="JWT"), "bearer"),
        "digest": (http_scheme(scheme="digest"), "digest"),
    }
    for name, (definition, http_scheme_value) in cases.items():
        doc = make_doc(
            schemes={name: definition},
            paths={"/pets": {"get": op(security=[{name: []}])}},
        )
        body = plan(doc, available={name: []}).json()
        inject = body["selected"]["schemes"][0]["inject"]
        assert inject["location"] == "authorization"
        assert inject["name"] == "Authorization"
        assert inject["http_scheme"] == http_scheme_value
        assert "Authorization" in inject["description"]


def test_oauth2_and_oidc_inject_bearer_authorization():
    doc = make_doc(
        schemes={"oauth": oauth2(), "oidc": oidc()},
        paths={"/pets": {"get": op(
            security=[{"oauth": ["pets:read"]}, {"oidc": ["openid"]}]
        )}},
    )
    body = plan(
        doc, available={"oauth": ["pets:read"], "oidc": ["openid"]}
    ).json()
    # Both are single-scheme items; canonical tie-break favors 'oauth'.
    inject = body["selected"]["schemes"][0]["inject"]
    assert inject["location"] == "authorization"
    assert inject["name"] == "Authorization"
    assert inject["http_scheme"] == "bearer"
    assert "Bearer" in inject["description"]


# --------------------------------------------------------------------------- #
# JSON Pointers
# --------------------------------------------------------------------------- #


def test_pointers_point_to_definition_and_occurrence():
    doc = make_doc(
        schemes={"apiKey": api_key()},
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    body = plan(doc, available={"apiKey": []}).json()
    scheme = body["selected"]["schemes"][0]
    assert scheme["scheme_pointer"] == "/components/securitySchemes/apiKey"
    assert (
        scheme["requirement_pointer"]
        == "/paths/~1pets/get/security/0/apiKey"
    )
    assert body["selected"]["item_pointer"] == "/paths/~1pets/get/security/0"
    assert body["alternatives"][0]["item_pointer"] == (
        "/paths/~1pets/get/security/0"
    )


def test_escaped_path_tokens_are_rfc6901_encoded():
    doc = make_doc(
        schemes={"apiKey": api_key()},
        paths={"/pets/{petId}": {"get": op(security=[{"apiKey": []}])}},
    )
    body = plan(doc, selector="GET /pets/{petId}",
               available={"apiKey": []}).json()
    pointer = body["selected"]["item_pointer"]
    assert pointer == "/paths/~1pets~1{petId}/get/security/0"


# --------------------------------------------------------------------------- #
# Security Scheme $ref resolution
# --------------------------------------------------------------------------- #


def test_local_security_scheme_ref_is_followed():
    doc = make_doc(
        components_extra={
            "schemas": {
                "ApiKeyScheme": api_key(key_name="X-Ref-Key"),
            }
        },
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    doc["components"]["securitySchemes"] = {
        "apiKey": {"$ref": "#/components/schemas/ApiKeyScheme"}
    }
    resp = plan(doc, available={"apiKey": []})
    assert resp.status_code == 200, resp.text
    scheme = resp.json()["selected"]["schemes"][0]
    assert scheme["scheme_pointer"] == "/components/schemas/ApiKeyScheme"
    assert scheme["inject"]["name"] == "X-Ref-Key"


def test_multihop_ref_chain_resolves_to_terminal():
    doc = make_doc(paths={"/pets": {"get": op(security=[{"apiKey": []}])}})
    doc["components"] = {
        "securitySchemes": {
            "apiKey": {"$ref": "#/components/x/A"},
        },
        "x": {
            "A": {"$ref": "#/components/x/B"},
            "B": api_key(key_name="X-Chain"),
        },
    }
    resp = plan(doc, available={"apiKey": []})
    assert resp.status_code == 200, resp.text
    scheme = resp.json()["selected"]["schemes"][0]
    assert scheme["scheme_pointer"] == "/components/x/B"
    assert scheme["inject"]["name"] == "X-Chain"


def test_31_sibling_fields_merge_over_ref_target():
    doc = make_doc(
        openapi="3.1.0",
        components_extra={
            "schemas": {"BaseKey": {"type": "apiKey", "in": "query",
                                     "name": "base"}},
        },
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    doc["components"]["securitySchemes"] = {
        "apiKey": {
            "$ref": "#/components/schemas/BaseKey",
            "in": "header",
        }
    }
    body = plan(doc, available={"apiKey": []}).json()
    inject = body["selected"]["schemes"][0]["inject"]
    # The nearer sibling overrides the referenced target; 'name' is kept.
    assert inject["location"] == "header"
    assert inject["name"] == "base"


def test_30_ref_siblings_are_ignored():
    doc = make_doc(
        openapi="3.0.3",
        components_extra={
            "schemas": {"BaseKey": {"type": "apiKey", "in": "query",
                                     "name": "base"}},
        },
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    doc["components"]["securitySchemes"] = {
        "apiKey": {
            "$ref": "#/components/schemas/BaseKey",
            "in": "header",
        }
    }
    body = plan(doc, available={"apiKey": []}).json()
    inject = body["selected"]["schemes"][0]["inject"]
    assert inject["location"] == "query"
    assert inject["name"] == "base"


def test_dangling_scheme_ref_is_422_without_secret_echo():
    doc = make_doc(paths={"/pets": {"get": op(security=[{"bad": []}])}})
    doc["components"] = {
        "securitySchemes": {
            "bad": {"$ref": "#/components/securitySchemes/missing"}
        }
    }
    resp = plan(doc, available={"bad": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "security_planning_failed"
    error = body["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert error["scheme"] == "bad"
    assert error["pointer"] == "/components/securitySchemes/bad"
    assert error["ref"] == "#/components/securitySchemes/missing"


def test_external_scheme_ref_is_422():
    doc = make_doc(paths={"/pets": {"get": op(security=[{"bad": []}])}})
    doc["components"] = {
        "securitySchemes": {
            "bad": {"$ref": "https://example.com/schemes.yaml#/Bad"}
        }
    }
    resp = plan(doc, available={"bad": []})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert error["pointer"] == "/components/securitySchemes/bad"
    assert "non-local" in error["error"]


def test_circular_scheme_ref_is_422():
    doc = make_doc(paths={"/pets": {"get": op(security=[{"a": []}])}})
    doc["components"] = {
        "securitySchemes": {
            "a": {"$ref": "#/components/securitySchemes/b"},
            "b": {"$ref": "#/components/securitySchemes/a"},
        }
    }
    resp = plan(doc, available={"a": []})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert "circular reference" in error["error"]


def test_ref_to_non_object_is_422():
    doc = make_doc(paths={"/pets": {"get": op(security=[{"bad": []}])}})
    doc["components"] = {
        "securitySchemes": {"bad": {"$ref": "#/info/version"}}
    }
    resp = plan(doc, available={"bad": []})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["code"] == "unresolvable_reference"


def test_unreachable_bad_scheme_does_not_block():
    # A broken scheme only referenced by an alternative that is itself
    # valid structurally must still validate (definitions are global); but
    # a broken scheme referenced *nowhere* by the effective security never
    # blocks -- here the operation overrides root security.
    doc = make_doc(
        schemes={
            "good": api_key(),
            "broken": {"type": "totally-unknown"},
        },
        security=[{"broken": []}],
        paths={"/pets": {"get": op(security=[{"good": []}])}},
    )
    resp = plan(doc, available={"good": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["selected"]["schemes"][0]["name"] == "good"


# --------------------------------------------------------------------------- #
# Definition validation (422)
# --------------------------------------------------------------------------- #


def test_unknown_scheme_name_in_requirement_is_422():
    doc = make_doc(
        schemes={"good": api_key()},
        paths={"/pets": {"get": op(security=[{"ghost": []}])}},
    )
    # The client does not claim to have 'ghost' available; the requirement
    # itself referencing an undeclared name is a contract defect (422).
    resp = plan(doc, available={})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "unknown_scheme"
    assert error["pointer"] == "/paths/~1pets/get/security/0/ghost"


def test_unsupported_scheme_type_is_422():
    doc = make_doc(
        schemes={"m": {"type": "mutualTLS"}},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert any(e["code"] == "invalid_security_scheme"
               and e["pointer"].endswith("/type") for e in errors)


def test_missing_type_is_422():
    doc = make_doc(
        schemes={"m": {"in": "header", "name": "X"}},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["code"] == "invalid_security_scheme"


def test_non_object_scheme_definition_is_422():
    for bad in (None, 42, "string", ["x"]):
        doc = make_doc(
            schemes={"m": bad},
            paths={"/pets": {"get": op(security=[{"m": []}])}},
        )
        resp = plan(doc, available={"m": []})
        assert resp.status_code == 422, bad
        error = resp.json()["errors"][0]
        assert error["code"] == "invalid_security_scheme"
        assert error["pointer"] == "/components/securitySchemes/m"


def test_non_object_path_item_is_422():
    doc = {
        "openapi": "3.0.3",
        "info": {"title": "X", "version": "1"},
        "paths": {"/pets": "not-an-object"},
    }
    resp = plan(doc)
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "invalid_security"
    assert error["pointer"] == "/paths/~1pets"


def test_apikey_definition_errors():
    cases = [
        {"type": "apiKey", "in": "body", "name": "X"},
        {"type": "apiKey", "in": "header"},
        {"type": "apiKey", "in": "header", "name": ""},
    ]
    for definition in cases:
        doc = make_doc(
            schemes={"m": definition},
            paths={"/pets": {"get": op(security=[{"m": []}])}},
        )
        resp = plan(doc, available={"m": []})
        assert resp.status_code == 422, definition
        assert resp.json()["errors"][0]["code"] == "invalid_security_scheme"


def test_http_definition_errors():
    bad_bearer_format = {
        "type": "http", "scheme": "bearer", "bearerFormat": 5,
    }
    doc = make_doc(
        schemes={"m": bad_bearer_format},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    assert plan(doc, available={"m": []}).status_code == 422

    doc["components"]["securitySchemes"]["m"] = {"type": "http"}
    assert plan(doc, available={"m": []}).status_code == 422


def test_openidconnect_requires_url():
    doc = make_doc(
        schemes={"m": {"type": "openIdConnect"}},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["pointer"].endswith("/openIdConnectUrl")


def test_oauth2_requires_flows_with_a_known_flow():
    doc = make_doc(
        schemes={"m": {"type": "oauth2", "flows": {}}},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    assert plan(doc, available={"m": []}).status_code == 422

    doc["components"]["securitySchemes"]["m"] = {"type": "oauth2"}
    assert plan(doc, available={"m": []}).status_code == 422


def test_oauth2_flow_url_requirements():
    # implicit needs authorizationUrl
    bad_implicit = {
        "implicit": {"scopes": {"a": "A"}}
    }
    doc = make_doc(
        schemes={"m": {"type": "oauth2", "flows": bad_implicit}},
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    pointer = resp.json()["errors"][0]["pointer"]
    assert pointer.endswith("/flows/implicit/authorizationUrl")

    # password needs tokenUrl
    doc["components"]["securitySchemes"]["m"] = {
        "type": "oauth2",
        "flows": {"password": {"scopes": {"a": "A"}}},
    }
    assert plan(doc, available={"m": []}).status_code == 422

    # valid implicit passes
    doc["components"]["securitySchemes"]["m"] = {
        "type": "oauth2",
        "flows": {
            "implicit": {
                "authorizationUrl": "https://idp/auth",
                "scopes": {"a": "A"},
            }
        },
    }
    assert plan(doc, available={"m": []}).status_code == 200


def test_oauth2_scopes_must_be_an_object_of_strings():
    doc = make_doc(
        schemes={
            "m": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://idp/token",
                        "scopes": {"a": 1},
                    }
                },
            }
        },
        paths={"/pets": {"get": op(security=[{"m": []}])}},
    )
    assert plan(doc, available={"m": []}).status_code == 422


def test_required_scope_not_declared_by_flow_is_422():
    doc = make_doc(
        schemes={
            "m": {
                "type": "oauth2",
                "flows": {
                    "implicit": {
                        "authorizationUrl": "https://idp/auth",
                        "scopes": {"a": "A"},
                    }
                },
            }
        },
        paths={"/pets": {"get": op(security=[{"m": ["ghost-scope"]}])}},
    )
    resp = plan(doc, available={"m": ["ghost-scope"]})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "undeclared_scope"
    assert error["scope"] == "ghost-scope"
    assert error["pointer"] == "/paths/~1pets/get/security/0/m"
    assert error["scheme_pointer"] == "/components/securitySchemes/m"


def test_scopes_on_non_oauth_scheme_are_422():
    doc = make_doc(
        schemes={"m": http_scheme()},
        paths={"/pets": {"get": op(security=[{"m": ["nope"]}])}},
    )
    resp = plan(doc, available={"m": ["nope"]})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "unexpected_scope"
    assert error["pointer"] == "/paths/~1pets/get/security/0/m"


def test_malformed_security_array_member_is_422():
    doc = make_doc(
        schemes={"m": api_key()},
        paths={"/pets": {"get": op(security=[{"m": []}, "not-an-object"])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "invalid_security"
    assert error["pointer"] == "/paths/~1pets/get/security/1"


def test_malformed_scope_list_is_422():
    doc = make_doc(
        schemes={"m": oauth2()},
        paths={"/pets": {"get": op(security=[{"m": "not-a-list"}])}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "invalid_security"
    assert error["pointer"] == "/paths/~1pets/get/security/0/m"


def test_non_array_security_is_422():
    doc = make_doc(
        schemes={"m": api_key()},
        paths={"/pets": {"get": op(security={"m": []})}},
    )
    resp = plan(doc, available={"m": []})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["pointer"] == (
        "/paths/~1pets/get/security"
    )


def test_non_array_inherited_root_security_is_422():
    doc = make_doc(schemes={"m": api_key()}, security={"m": []})
    resp = plan(doc)
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["pointer"] == "/security"


def test_non_object_operation_is_422():
    doc = {
        "openapi": "3.0.3",
        "info": {"title": "X", "version": "1"},
        "paths": {"/pets": {"get": "not-an-object"}},
    }
    resp = plan(doc)
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["code"] == "invalid_security"
    assert error["pointer"] == "/paths/~1pets/get"


def test_errors_are_stably_sorted():
    doc = make_doc(
        schemes={
            "zbad": {"type": "nope"},
            "abad": {"type": "nope"},
        },
        paths={"/pets": {"get": op(
            security=[{"zbad": []}, {"abad": []}]
        )}},
    )
    resp = plan(doc, available={"zbad": [], "abad": []})
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    keys = [(e["code"], e["pointer"]) for e in errors]
    assert keys == sorted(keys)


def test_path_item_ref_is_resolved_before_operation_lookup():
    doc = make_doc(
        schemes={"apiKey": api_key()},
        components_extra={
            "pathItems": {
                "Pets": {
                    "get": op(security=[{"apiKey": []}]),
                }
            }
        },
        openapi="3.1.0",
        paths={"/pets": {"$ref": "#/components/pathItems/Pets"}},
    )
    resp = plan(doc, available={"apiKey": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["selected"]["schemes"][0]["name"] == "apiKey"
    # The requirement pointer stays anchored at the path occurrence (the
    # path item was resolved through its $ref chain first).
    assert body["selected"]["item_pointer"] == "/paths/~1pets/get/security/0"


# --------------------------------------------------------------------------- #
# 400 errors
# --------------------------------------------------------------------------- #


def test_bad_document_shape_is_400():
    resp = client.post(
        "/security/plan",
        json={"document": {"openapi": "2.0", "paths": {}}, "selector": "GET /x"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_openapi_document"


def test_malformed_selector_is_400():
    doc = make_doc()
    resp = plan(doc, selector="garbage")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_selector"


def test_unknown_path_and_method_are_400():
    doc = make_doc()
    assert plan(doc, selector="GET /nope").status_code == 400
    assert plan(doc, selector="DELETE /pets").status_code == 400


def test_available_must_be_an_object_of_string_arrays():
    doc = make_doc(schemes={"m": api_key()})
    resp = plan(doc, available={"m": "not-a-list"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_security_request"

    resp2 = plan(doc, available={"m": [1, 2]})
    assert resp2.status_code == 400


def test_available_unknown_scheme_is_400():
    doc = make_doc(schemes={"m": api_key()})
    resp = plan(doc, available={"ghost": []})
    assert resp.status_code == 400
    messages = resp.json()["messages"]
    assert any("ghost" in m for m in messages)


def test_secret_fields_are_rejected_without_echoing_values():
    doc = make_doc(schemes={"m": api_key()})
    for field_name in (
        "token", "secret", "password", "credentials",
        "client_secret", "api_key",
    ):
        resp = plan(doc, available={"m": []}, **{field_name: "SUPERSECRET"})
        assert resp.status_code == 400, field_name
        body = resp.json()
        assert body["error"] == "invalid_security_request"
        assert "SUPERSECRET" not in str(body)


def test_request_body_must_be_object():
    # FastAPI body-shape validation rejects a non-object payload with 422,
    # consistent with the other endpoints.
    resp = client.post("/security/plan", json=["not", "an", "object"])
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Stability and non-mutation
# --------------------------------------------------------------------------- #


def test_response_is_independent_of_input_key_order():
    schemes = {
        "a": api_key(key_name="A"),
        "b": api_key(key_name="B"),
        "oauth": oauth2(),
    }
    # AND item (a,b) vs single-scheme oauth item; all available.
    paths1 = {"/pets": {"get": op(security=[
        {"a": [], "b": []},
        {"oauth": ["pets:read"]},
    ])}}
    doc1 = make_doc(schemes=schemes, paths=paths1)
    doc2 = copy.deepcopy(doc1)
    doc2["paths"]["/pets"]["get"]["security"] = list(
        reversed(doc2["paths"]["/pets"]["get"]["security"])
    )

    body1 = plan(
        doc1, available={"a": [], "b": [], "oauth": ["pets:read", "pets:write"]}
    ).json()
    body2 = plan(
        doc2, available={"a": [], "b": [], "oauth": ["pets:write", "pets:read"]}
    ).json()
    # The single-scheme oauth item wins in both documents.
    assert body1["selected"]["schemes"][0]["name"] == "oauth"
    assert body2["selected"]["schemes"][0]["name"] == "oauth"


def test_input_document_is_not_mutated():
    doc = make_doc(
        schemes={
            "apiKey": {"$ref": "#/components/schemas/Key"},
        },
        components_extra={"schemas": {"Key": api_key(key_name="X")}},
        openapi="3.1.0",
        paths={"/pets": {"get": op(security=[{"apiKey": []}])}},
    )
    snapshot = copy.deepcopy(doc)
    plan(doc, available={"apiKey": []})
    assert doc == snapshot


def test_method_selector_is_case_insensitive():
    doc = make_doc(schemes={"m": api_key()},
                   paths={"/pets": {"get": op(security=[{"m": []}])}})
    resp = plan(doc, selector="get /pets", available={"m": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["selector"] == "GET /pets"


def test_no_available_means_nothing_satisfiable_but_still_200():
    doc = make_doc(
        schemes={"a": api_key(), "b": http_scheme()},
        paths={"/pets": {"get": op(
            security=[{"a": []}, {"b": []}]
        )}},
    )
    resp = plan(doc)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["satisfiable"] is False
    assert body["selected"] is None
    assert all(a["satisfiable"] is False for a in body["alternatives"])
    missing_names = {
        a["item_index"]: [m["scheme"] for m in a["missing"]]
        for a in body["alternatives"]
    }
    assert missing_names == {0: ["a"], 1: ["b"]}


def test_multiple_missing_scopes_are_sorted():
    doc = make_doc(
        schemes={"oauth": oauth2()},
        paths={"/pets": {"get": op(security=[
            {"oauth": ["pets:write", "pets:read", "admin"]}
        ])}},
    )
    body = plan(doc, available={"oauth": []}).json()
    missing = body["alternatives"][0]["missing"]
    assert [m["scope"] for m in missing] == [
        "admin", "pets:read", "pets:write"
    ]


def test_shared_scheme_across_alternatives_validated_once():
    # Same oauth2 scheme in two items with different scope requirements;
    # both requirements are evaluated against grants.
    doc = make_doc(
        schemes={"oauth": oauth2()},
        paths={"/pets": {"get": op(security=[
            {"oauth": ["pets:read"]},
            {"oauth": ["pets:write"]},
        ])}},
    )
    body = plan(doc, available={"oauth": ["pets:write"]}).json()
    assert body["satisfiable"] is True
    assert body["selected"]["item_index"] == 1
    alt0, alt1 = body["alternatives"]
    assert alt0["satisfiable"] is False
    assert alt0["missing"][0]["scope"] == "pets:read"
    assert alt1["satisfiable"] is True


def test_root_service_listing_advertises_endpoint():
    resp = client.get("/")
    assert resp.json()["security_plan"] == "POST /security/plan"
