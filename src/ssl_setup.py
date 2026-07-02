def enable_os_trust_store():
    """Use the OS certificate store (handles corporate SSL-inspection proxies).

    On machines behind an SSL-intercepting proxy, Python's bundled CA list does
    not trust the proxy's re-signing certificate. truststore delegates to the
    operating system's trust store, which does. No-op if truststore is absent.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
