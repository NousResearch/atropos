#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=${1:-${WORKSPACE_ROOT:-/workspace}}
REPO_PATH=${TASK_REPO_PATH:-$WORKSPACE_ROOT/workspace}
REPO_URL=${TASK_REPO_URL:-}
REPO_BRANCH=${TASK_REPO_BRANCH:-}
REPO_REV=${TASK_REPO_REV:-}
LANG=${TASK_LANGUAGE:-}

log() {
  printf '[bootstrap] %s\n' "$*"
}

clone_repo() {
  if [[ -z "$REPO_URL" ]]; then
    log "TASK_REPO_URL is required" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$REPO_PATH")"

  if [[ -d "$REPO_PATH/.git" ]]; then
    log "Updating existing repo at $REPO_PATH"
    git -C "$REPO_PATH" fetch --all --prune
  else
    log "Cloning $REPO_URL into $REPO_PATH"
    git clone "$REPO_URL" "$REPO_PATH"
  fi

  pushd "$REPO_PATH" >/dev/null
  git reset --hard
  if [[ -n "$REPO_REV" ]]; then
    git checkout "$REPO_REV"
  elif [[ -n "$REPO_BRANCH" ]]; then
    git checkout "$REPO_BRANCH" || git checkout -b "$REPO_BRANCH" origin/"$REPO_BRANCH"
  else
    git checkout "$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')"
  fi
  git clean -fdx
  popd >/dev/null
}

run_python_bootstrap() {
  command -v pip >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt || true
  fi
  if [[ -f pyproject.toml ]]; then
    pip install . || true
  fi
  popd >/dev/null
}

run_node_bootstrap() {
  command -v npm >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f package.json ]]; then
    npm install || true
  fi
  popd >/dev/null
}

run_rust_bootstrap() {
  command -v cargo >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f Cargo.toml ]]; then
    cargo fetch || true
  fi
  popd >/dev/null
}

run_go_bootstrap() {
  command -v go >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f go.mod ]]; then
    go mod download || true
  fi
  popd >/dev/null
}

run_java_bootstrap() {
  pushd "$REPO_PATH" >/dev/null
  if [[ -x ./gradlew ]]; then
    ./gradlew --no-daemon classes || true
  elif [[ -x ./mvnw ]]; then
    ./mvnw -B dependency:go-offline || true
  elif command -v mvn >/dev/null && [[ -f pom.xml ]]; then
    mvn -B dependency:go-offline || true
  fi
  popd >/dev/null
}

run_csharp_bootstrap() {
  command -v dotnet >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if ls *.sln *.csproj >/dev/null 2>&1; then
    dotnet restore || true
  fi
  popd >/dev/null
}

run_php_bootstrap() {
  command -v composer >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f composer.json ]]; then
    composer install --no-interaction || true
  fi
  popd >/dev/null
}

run_ruby_bootstrap() {
  command -v bundle >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f Gemfile ]]; then
    bundle install || true
  fi
  popd >/dev/null
}

run_dart_bootstrap() {
  command -v dart >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if ls pubspec.* >/dev/null 2>&1; then
    dart pub get || true
  fi
  popd >/dev/null
}

run_elixir_bootstrap() {
  command -v mix >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if [[ -f mix.exs ]]; then
    mix deps.get || true
  fi
  popd >/dev/null
}

run_lua_bootstrap() {
  command -v luarocks >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if compgen -G "*.rockspec" >/dev/null; then
    luarocks install --deps-only *.rockspec || true
  fi
  popd >/dev/null
}

run_haskell_bootstrap() {
  command -v cabal >/dev/null || return 0
  pushd "$REPO_PATH" >/dev/null
  if compgen -G "*.cabal" >/dev/null; then
    cabal update || true
  fi
  popd >/dev/null
}

run_bootstrap_steps() {
  case "$LANG" in
    Python|"Jupyter Notebook")
      run_python_bootstrap
      ;;
    TypeScript|JavaScript|HTML)
      run_node_bootstrap
      ;;
    Rust)
      run_rust_bootstrap
      ;;
    Go)
      run_go_bootstrap
      ;;
    Java|Kotlin|Scala)
      run_java_bootstrap
      ;;
    "C#"|"CSharp")
      run_csharp_bootstrap
      ;;
    PHP)
      run_php_bootstrap
      ;;
    Ruby)
      run_ruby_bootstrap
      ;;
    Dart)
      run_dart_bootstrap
      ;;
    Elixir)
      run_elixir_bootstrap
      ;;
    Lua)
      run_lua_bootstrap
      ;;
    Haskell)
      run_haskell_bootstrap
      ;;
    *)
      log "No additional bootstrap steps for $LANG"
      ;;
  esac
}

write_metadata() {
  cat >"$WORKSPACE_ROOT/cline-workspace.json" <<EOF
{
  "language": "$LANG",
  "repo_path": "$REPO_PATH",
  "repo_url": "$REPO_URL"
}
EOF
}

main() {
  log "Preparing workspace at $WORKSPACE_ROOT for language '$LANG'"
  clone_repo
  run_bootstrap_steps
  write_metadata
  log "Workspace ready at $REPO_PATH"
}

main "$@"
