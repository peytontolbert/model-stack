#!/usr/bin/env bash
# Plan or resume a bounded raw-model cache from /arxiv to /data.
set -euo pipefail

SOURCE=/arxiv/models
DESTINATION=/data/models
MIN_GIB=8
MAX_GIB=320
COPY=0
ALL_HEAVY=0
EXCLUDES='^(HunyuanVideo-Avatar|assets|examples|figures|images|inputs|outputs|\.cache|\.ipynb_checkpoints|repository_library)$'
INCLUDES=()

usage() {
  printf '%s\n' 'Usage: preload_heavy_models_to_data.sh [--copy] [--all-heavy] [--include NAME] [--min-gib N] [--max-gib N]'
}

while (($#)); do
  case "$1" in
    --copy) COPY=1 ;;
    --all-heavy) ALL_HEAVY=1 ;;
    --include) INCLUDES+=("$2"); shift ;;
    --min-gib) MIN_GIB="$2"; shift ;;
    --max-gib) MAX_GIB="$2"; shift ;;
    --source) SOURCE="$2"; shift ;;
    --destination) DESTINATION="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -d "$SOURCE" ]] || { printf 'Missing source: %s\n' "$SOURCE" >&2; exit 1; }
[[ -d "$DESTINATION" ]] || { printf 'Missing destination: %s\n' "$DESTINATION" >&2; exit 1; }

declare -A requested=()
for name in "${INCLUDES[@]}"; do requested["$name"]=1; done
declare -a candidates=()
while IFS= read -r -d '' path; do
  name=$(basename "$path")
  [[ "$name" =~ $EXCLUDES ]] && continue
  [[ $ALL_HEAVY -eq 1 || ${requested[$name]+yes} ]] || continue
  bytes=$(du -sB1 -- "$path" | awk '{print $1}')
  gib=$(( (bytes + 1073741823) / 1073741824 ))
  if [[ ${requested[$name]+yes} || $gib -ge $MIN_GIB ]]; then
    candidates+=("$bytes:$name")
  fi
done < <(find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -print0)

((${#candidates[@]})) || { printf '%s\n' 'No models selected. Use --all-heavy or --include NAME.'; exit 0; }
IFS=$'\n' candidates=($(sort -nr <<<"${candidates[*]}")); unset IFS

selected_bytes=0
declare -a selected=()
printf '%-58s %10s %s\n' 'MODEL' 'SIZE (GiB)' 'CACHE STATUS'
for item in "${candidates[@]}"; do
  bytes=${item%%:*}; name=${item#*:}; gib=$(( (bytes + 1073741823) / 1073741824 ))
  if (( selected_bytes + bytes > MAX_GIB * 1073741824 )); then
    printf '%-58s %10d %s\n' "$name" "$gib" 'SKIPPED: budget'
    continue
  fi
  selected_bytes=$(( selected_bytes + bytes )); selected+=("$item")
  [[ -d "$DESTINATION/$name" ]] && status='existing/resumable' || status='new'
  printf '%-58s %10d %s\n' "$name" "$gib" "$status"
done

selected_gib=$(( (selected_bytes + 1073741823) / 1073741824 ))
available_gib=$(df -BG --output=avail "$DESTINATION" | tail -n 1 | tr -dc '0-9')
printf '\nSelected source size: %d GiB; destination free space: %d GiB.\n' "$selected_gib" "$available_gib"
(( selected_gib <= available_gib )) || { printf '%s\n' 'Selection exceeds free space.' >&2; exit 1; }
(( COPY == 1 )) || exit 0

mkdir -p "$DESTINATION/.preload-manifests"
manifest="$DESTINATION/.preload-manifests/preload-$(date +%Y%m%d-%H%M%S).tsv"
printf 'model\tsource_bytes\n' > "$manifest"
for item in "${selected[@]}"; do
  bytes=${item%%:*}; name=${item#*:}
  printf 'Caching %s\n' "$name"
  rsync -aH --partial --append-verify --info=progress2 -- "$SOURCE/$name/" "$DESTINATION/$name/"
  printf '%s\t%s\n' "$name" "$bytes" >> "$manifest"
done
printf 'Completed cache manifest: %s\n' "$manifest"
