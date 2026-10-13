#!/usr/bin/env bash
# Open a MySQL client using the same environment variables as training.
#   export DB_HOST=127.0.0.1 DB_USER=root DB_PASSWORD=... DB_DATABASE=dart
mysql -h "${DB_HOST:-127.0.0.1}" -P "${DB_PORT:-3306}" -u "${DB_USER:-root}" -p"${DB_PASSWORD:?Set DB_PASSWORD}" "${DB_DATABASE:-dart}"
