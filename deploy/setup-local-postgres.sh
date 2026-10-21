#!/usr/bin/env bash
# Local PostGIS for development and verification (Ubuntu / WSL).
# Production uses the postgis container from docker-compose.yml instead.
set -euo pipefail
DB_PASSWORD="${1:?usage: setup-local-postgres.sh <password>}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq postgresql postgresql-postgis >/dev/null
service postgresql start
sleep 3
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
do \$\$ begin
  if not exists (select from pg_roles where rolname = 'urbantrace') then
    create role urbantrace login password '${DB_PASSWORD}';
  else
    alter role urbantrace password '${DB_PASSWORD}';
  end if;
end \$\$;
SQL
sudo -u postgres psql -tc "select 1 from pg_database where datname='urbantrace'" | grep -q 1 \
  || sudo -u postgres createdb -O urbantrace urbantrace
sudo -u postgres psql -d urbantrace -c "create extension if not exists postgis; create extension if not exists pgcrypto;"
echo "ready: postgresql://urbantrace:***@localhost:5432/urbantrace"
