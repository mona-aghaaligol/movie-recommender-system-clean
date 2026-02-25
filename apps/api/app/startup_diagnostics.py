"""
Startup diagnostics module for ECS container debugging.

This module runs at container start (before uvicorn) and prints:

- Python version
- OpenSSL version
- CA bundle presence
- /etc/resolv.conf contents
- Safe Mongo URI details (scheme + host only)
- DNS resolution checks
- TLS connectivity tests (public + Atlas)

IMPORTANT:
- Never prints raw credentials.
- Never prints full connection string.
"""

import os
import sys
import ssl
import socket
import datetime
import re
import traceback


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _now_utc():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _safe_print(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _mask_mongo_uri(uri: str) -> str:
    """
    Masks credentials in MongoDB URI.
    Example:
    mongodb+srv://user:pass@cluster.mongodb.net
    becomes:
    mongodb+srv://***@cluster.mongodb.net
    """
    return re.sub(r"//([^@/]+)@", "//***@", uri)


def _extract_scheme(uri: str) -> str:
    if "://" in uri:
        return uri.split("://", 1)[0]
    return "unknown"


def _extract_host(uri: str) -> str:
    # Extract host after credentials (if present)
    return uri.split("@")[-1].split("/", 1)[0]


# ---------------------------------------------------------------------
# Version diagnostics
# ---------------------------------------------------------------------

_safe_print(f"[diag] timestamp_utc={_now_utc()}")
_safe_print(f"[diag] python_version={sys.version.replace(chr(10), ' ')}")
_safe_print(f"[diag] openssl_version={ssl.OPENSSL_VERSION}")
_safe_print(f"[diag] ssl_default_verify_paths={ssl.get_default_verify_paths()}")


# ---------------------------------------------------------------------
# CA bundle check
# ---------------------------------------------------------------------

CA_PATHS = [
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/ssl/cert.pem",
    "/etc/pki/tls/certs/ca-bundle.crt",
]

for path in CA_PATHS:
    _safe_print(f"[diag] ca_bundle_exists path={path} exists={os.path.exists(path)}")


# ---------------------------------------------------------------------
# resolv.conf check
# ---------------------------------------------------------------------

_safe_print("[diag] resolv_conf_begin")
try:
    with open("/etc/resolv.conf", "r") as f:
        for line in f:
            _safe_print(f"[diag] resolv_conf_line {line.strip()}")
except Exception as e:
    _safe_print(f"[diag] resolv_conf_error {repr(e)}")
_safe_print("[diag] resolv_conf_end")


# ---------------------------------------------------------------------
# Mongo URI (safe)
# ---------------------------------------------------------------------

mongo_uri = os.environ.get("MONGO_URI_DEV", "")

if not mongo_uri:
    _safe_print("[diag] mongo_uri_env name=MONGO_URI_DEV status=missing")
else:
    safe_uri = _mask_mongo_uri(mongo_uri)
    scheme = _extract_scheme(mongo_uri)
    host = _extract_host(mongo_uri)

    _safe_print(
        f"[diag] mongo_uri_env name=MONGO_URI_DEV status=present "
        f"scheme={scheme} host={host}"
    )
    _safe_print(
        f"[diag] mongo_uri_env_safe_prefix "
        f"{safe_uri[:140]}{'...' if len(safe_uri) > 140 else ''}"
    )


# ---------------------------------------------------------------------
# DNS resolution helper
# ---------------------------------------------------------------------

def _dns_resolution(host: str):
    try:
        result = socket.getaddrinfo(host, None)
        _safe_print(f"[diag] dns_resolution host={host} result={result}")
    except Exception as e:
        _safe_print(f"[diag] dns_resolution host={host} error={repr(e)}")


# ---------------------------------------------------------------------
# TLS test helper (hostname -> connect(host), SNI=host)
# ---------------------------------------------------------------------

def _tls_test(host: str, port: int, label: str):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                _safe_print(
                    f"[diag] {label} host={host} port={port} "
                    f"result={{'status': 'ok', "
                    f"'tls_version': '{ssock.version()}', "
                    f"'cipher': {ssock.cipher()}, "
                    f"'peer_cert_subject': {cert.get('subject')}, "
                    f"'peer_cert_issuer': {cert.get('issuer')}}}"
                )
    except Exception as e:
        _safe_print(
            f"[diag] {label} host={host} port={port} "
            f"result={{'status': 'fail', 'error': {repr(e)}}}"
        )


# ---------------------------------------------------------------------
# TLS test helper (IP connect, but SNI=hostname)
# ---------------------------------------------------------------------

def _tls_test_ip(ip: str, sni_host: str, port: int, label: str):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((ip, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=sni_host) as ssock:
                cert = ssock.getpeercert()
                _safe_print(
                    f"[diag] {label} ip={ip} sni={sni_host} port={port} "
                    f"result={{'status': 'ok', "
                    f"'tls_version': '{ssock.version()}', "
                    f"'cipher': {ssock.cipher()}, "
                    f"'peer_cert_subject': {cert.get('subject')}, "
                    f"'peer_cert_issuer': {cert.get('issuer')}}}"
                )
    except Exception as e:
        _safe_print(
            f"[diag] {label} ip={ip} sni={sni_host} port={port} "
            f"result={{'status': 'fail', 'error': {repr(e)}}}"
        )


# ---------------------------------------------------------------------
# Public TLS test (control test)
# ---------------------------------------------------------------------

_dns_resolution("www.google.com")
_tls_test("www.google.com", 443, "tls_public")


# ---------------------------------------------------------------------
# Atlas TLS test (SRV resolve via PyMongo, then test each shard)
# + IP+SNI test for shard-00-02
# ---------------------------------------------------------------------

if mongo_uri:
    try:
        # Important: mongodb+srv host itself may not have A/AAAA records.
        # PyMongo resolves SRV to concrete shard hostnames; we test TLS against those.
        seed_host = _extract_host(mongo_uri)
        _safe_print(f"[diag] atlas_srv_seed_host={seed_host}")

        try:
            from pymongo import uri_parser as _uri_parser
        except Exception as e:
            _safe_print(f"[diag] atlas_srv_resolve_error import_pymongo_uri_parser {repr(e)}")
            raise

        parsed = _uri_parser.parse_uri(mongo_uri, warn=False)
        nodelist = parsed.get("nodelist") or []

        _safe_print(f"[diag] atlas_srv_nodelist={nodelist}")

        for (h, p) in nodelist:
            port = int(p or 27017)
            _dns_resolution(h)
            _tls_test(h, port, "tls_atlas_shard")

            # Focused IP+SNI test (kept minimal to avoid noisy logs)
            # If this succeeds but hostname TLS fails, it strongly suggests DNS/egress path interference.
            if h == "ac-f6c804o-shard-00-02.tjh5xg8.mongodb.net":
                try:
                    infos = socket.getaddrinfo(h, port)
                    ip = infos[0][4][0] if infos else None
                except Exception:
                    ip = None

                if ip:
                    _tls_test_ip(ip, h, port, "tls_atlas_ip_sni")
                else:
                    _safe_print(f"[diag] tls_atlas_ip_sni_skip reason=no_ip host={h}")

    except Exception:
        _safe_print(f"[diag] atlas_test_error {traceback.format_exc()}")
