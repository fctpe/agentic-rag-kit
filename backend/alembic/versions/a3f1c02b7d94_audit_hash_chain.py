"""audit log hash chain

Revision ID: a3f1c02b7d94
Revises: 13e754f2ef3c
Create Date: 2026-08-03 11:04:18.552310

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.models.tables import AuditLog
from app.security.audit import chain_hash

revision: str = 'a3f1c02b7d94'
down_revision: str | None = '13e754f2ef3c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('audit_log', sa.Column('prev_hash', sa.String(length=64), nullable=False, server_default=''))
    op.add_column('audit_log', sa.Column('entry_hash', sa.String(length=64), nullable=False, server_default=''))
    # Chain position. Added nullable so the existing rows can be numbered by the
    # backfill below; it becomes the primary key and gets its sequence after.
    op.add_column('audit_log', sa.Column('seq', sa.BigInteger(), nullable=True))

    # Entries written before this migration were never hashed at write time, so
    # the chain is built over them now. That protects them from here on; it says
    # nothing about whether they were altered before this ran.
    connection = op.get_bind()
    rows = connection.execute(sa.text(
        'SELECT id, user_id, action, resource, detail, created_at FROM audit_log '
        'ORDER BY created_at, id'
    )).all()
    previous = ''
    for position, row in enumerate(rows, start=1):
        entry = AuditLog(
            id=row.id,
            user_id=row.user_id,
            action=row.action,
            resource=row.resource,
            detail=row.detail,
            created_at=row.created_at,
            prev_hash=previous,
        )
        previous = chain_hash(entry)
        connection.execute(
            sa.text(
                'UPDATE audit_log SET seq = :seq, prev_hash = :prev, entry_hash = :entry '
                'WHERE id = :id'
            ),
            {'seq': position, 'prev': entry.prev_hash, 'entry': previous, 'id': row.id},
        )

    op.alter_column('audit_log', 'seq', existing_type=sa.BigInteger(), nullable=False)
    op.drop_constraint('audit_log_pkey', 'audit_log', type_='primary')
    op.create_primary_key('audit_log_pkey', 'audit_log', ['seq'])
    op.create_unique_constraint('uq_audit_log_id', 'audit_log', ['id'])
    # What BIGSERIAL expands to, spelled out because the column already exists.
    # From here the database hands out chain positions: ordering appends by a
    # Python clock and a uuid4 forked the chain under ordinary concurrency.
    op.execute('CREATE SEQUENCE audit_log_seq_seq OWNED BY audit_log.seq')
    op.execute("ALTER TABLE audit_log ALTER COLUMN seq SET DEFAULT nextval('audit_log_seq_seq')")
    op.execute(
        "SELECT setval('audit_log_seq_seq', COALESCE((SELECT MAX(seq) FROM audit_log), 0) + 1, false)"
    )

    # The default existed only to backfill; a future insert that skips the chain
    # must fail rather than land with empty hashes.
    op.alter_column('audit_log', 'prev_hash', server_default=None)
    op.alter_column('audit_log', 'entry_hash', server_default=None)


def downgrade() -> None:
    op.execute('ALTER TABLE audit_log ALTER COLUMN seq DROP DEFAULT')
    op.execute('DROP SEQUENCE audit_log_seq_seq')
    op.drop_constraint('uq_audit_log_id', 'audit_log', type_='unique')
    op.drop_constraint('audit_log_pkey', 'audit_log', type_='primary')
    op.create_primary_key('audit_log_pkey', 'audit_log', ['id'])
    op.drop_column('audit_log', 'seq')
    op.drop_column('audit_log', 'entry_hash')
    op.drop_column('audit_log', 'prev_hash')
