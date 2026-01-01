from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'client',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('slug', sa.String(), nullable=False, unique=True),
        sa.Column('business_name', sa.String(), nullable=False),
        sa.Column('inquiry_email', sa.String(), nullable=True),
        sa.Column('pricing', sa.Text(), nullable=True),
        sa.Column('business_description', sa.Text(), nullable=True),
        sa.Column('sign_off_name', sa.String(), nullable=True),
        sa.Column('mimic_email', sa.String(), nullable=True),
        sa.Column('slack_webhook_url', sa.String(), nullable=True),
        sa.Column('reply_mode', sa.String(), nullable=False, default='auto_send'),
    )
    op.create_index('ix_client_slug', 'client', ['slug'])

    op.create_table(
        'mailboxconnection',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('client_id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('connected_email', sa.String(), nullable=False),
        sa.Column('access_token_encrypted', sa.Text(), nullable=False),
        sa.Column('refresh_token_encrypted', sa.Text(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(), nullable=False, default='active'),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('provider_metadata', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['client_id'], ['client.id'], ),
    )
    op.create_index('ix_mailbox_client', 'mailboxconnection', ['client_id'])

    op.create_table(
        'emailthread',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('client_id', sa.Integer(), nullable=False),
        sa.Column('connection_id', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('provider_thread_id', sa.String(), nullable=False),
        sa.Column('subject', sa.String(), nullable=True),
        sa.Column('last_message_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['client_id'], ['client.id'], ),
        sa.ForeignKeyConstraint(['connection_id'], ['mailboxconnection.id'], ),
        sa.UniqueConstraint('client_id', 'provider_thread_id', name='uq_client_thread')
    )
    op.create_index('ix_thread_client_provider', 'emailthread', ['client_id', 'provider_thread_id'])

    op.create_table(
        'emailmessage',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('client_id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('provider_message_id', sa.String(), nullable=False),
        sa.Column('direction', sa.String(), nullable=False),
        sa.Column('from_email', sa.String(), nullable=False),
        sa.Column('to_email', sa.String(), nullable=False),
        sa.Column('cc_json', sa.Text(), nullable=True),
        sa.Column('subject', sa.String(), nullable=True),
        sa.Column('body_text', sa.Text(), nullable=True),
        sa.Column('snippet', sa.Text(), nullable=True),
        sa.Column('received_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['client_id'], ['client.id'], ),
        sa.ForeignKeyConstraint(['thread_id'], ['emailthread.id'], ),
        sa.UniqueConstraint('client_id', 'provider_message_id', name='uq_client_message')
    )
    op.create_index('ix_msg_client_provider_msg', 'emailmessage', ['client_id', 'provider_message_id'])

    op.create_table(
        'lead',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('client_id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.Integer(), nullable=True),
        sa.Column('lead_name', sa.String(), nullable=True),
        sa.Column('lead_email', sa.String(), nullable=True),
        sa.Column('lead_phone', sa.String(), nullable=True),
        sa.Column('message_text', sa.Text(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('email_status', sa.String(), nullable=False, default='pending'),
        sa.Column('slack_status', sa.String(), nullable=False, default='pending'),
        sa.Column('ai_status', sa.String(), nullable=False, default='pending'),
        sa.Column('error_email', sa.Text(), nullable=True),
        sa.Column('error_slack', sa.Text(), nullable=True),
        sa.Column('error_ai', sa.Text(), nullable=True),
        sa.Column('correlation_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['client_id'], ['client.id'], ),
        sa.ForeignKeyConstraint(['thread_id'], ['emailthread.id'], ),
    )
    op.create_index('ix_lead_client', 'lead', ['client_id'])

    op.create_table(
        'aidraft',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('lead_id', sa.Integer(), nullable=False),
        sa.Column('prompt_text', sa.Text(), nullable=False),
        sa.Column('draft_text', sa.Text(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('sent_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lead_id'], ['lead.id'], ),
    )
    op.create_table(
        'job',
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('job_type', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, default='pending'),
        sa.Column('attempts', sa.Integer(), nullable=False, default=0),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(), nullable=True),
    )


def downgrade():
    op.drop_table('aidraft')
    op.drop_table('lead')
    op.drop_table('emailmessage')
    op.drop_table('emailthread')
    op.drop_table('mailboxconnection')
    op.drop_table('client')
