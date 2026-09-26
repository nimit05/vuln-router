#!/bin/bash
# Stage Joern and a JDK on the LOGIN node.
#
# Compute nodes reach HTTPS only through a TLS-intercepting proxy, so anything
# downloaded has to be fetched here first -- the same constraint that forces
# model weights to be pre-staged. The cluster has no `java` and no `module`
# system, so the JDK comes as a portable tarball rather than a package.
#
# Joern needs Java 17+. Temurin 17 is the last LTS Joern's own installer targets.
set -uo pipefail
DEST="$HOME/tools"
mkdir -p "$DEST" "$HOME/logs"
LOG="$HOME/logs/stage_joern.log"

log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

log "staging into $DEST"

# --- JDK ------------------------------------------------------------------
if [ ! -x "$DEST/jdk/bin/java" ]; then
    log "downloading Temurin 17 JDK"
    curl -sL --max-time 900 -o "$DEST/jdk.tar.gz" \
      "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" \
      || { log "JDK download FAILED"; exit 1; }
    mkdir -p "$DEST/jdk"
    tar -xzf "$DEST/jdk.tar.gz" -C "$DEST/jdk" --strip-components=1 \
      || { log "JDK extract FAILED"; exit 1; }
    rm -f "$DEST/jdk.tar.gz"
fi
export JAVA_HOME="$DEST/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
log "java: $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"

# --- Joern ----------------------------------------------------------------
# The convenience path releases/latest/download/joern-cli.zip 302s to an asset
# name that no longer exists and lands on a 404 -- which curl -sL happily saves
# as a 9-byte file. Resolve the real asset through the API instead, and verify
# the checksum so a truncated 1.8 GB download fails loudly rather than at unzip.
if [ ! -x "$DEST/joern-cli/joern" ]; then
    log "resolving joern release asset"
    URL="$(curl -s --max-time 60 \
        "https://api.github.com/repos/joernio/joern/releases/latest" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)
for a in d.get('assets',[]):
    if a['name']=='joern-cli-linux-x86_64.zip':
        print(a['browser_download_url']); break
")"
    [ -n "$URL" ] || { log "could not resolve joern asset URL"; exit 1; }
    log "downloading $URL (~1.8 GB)"
    curl -L --max-time 3600 -C - -o "$DEST/joern-cli.zip" "$URL" \
      || { log "joern download FAILED"; exit 1; }
    SZ=$(stat -c%s "$DEST/joern-cli.zip" 2>/dev/null || echo 0)
    [ "$SZ" -gt 100000000 ] || { log "joern zip too small ($SZ bytes) -- FAILED"; exit 1; }
    curl -sL --max-time 120 -o "$DEST/joern-cli.zip.sha512" "$URL.sha512" || true
    if [ -s "$DEST/joern-cli.zip.sha512" ]; then
        EXP=$(awk "{print \$1}" "$DEST/joern-cli.zip.sha512")
        GOT=$(sha512sum "$DEST/joern-cli.zip" | awk "{print \$1}")
        [ "$EXP" = "$GOT" ] && log "checksum ok" || { log "CHECKSUM MISMATCH"; exit 1; }
    fi
    unzip -q -o "$DEST/joern-cli.zip" -d "$DEST" || { log "joern unzip FAILED"; exit 1; }
    rm -f "$DEST/joern-cli.zip" "$DEST/joern-cli.zip.sha512"
    chmod +x "$DEST/joern-cli/"*.sh "$DEST/joern-cli/joern"* 2>/dev/null
fi

log "joern files: $(ls "$DEST/joern-cli" | head -8 | tr '\n' ' ')"
log "c2cpg present: $([ -x "$DEST/joern-cli/c2cpg.sh" ] && echo yes || echo NO)"
log "staging complete"
