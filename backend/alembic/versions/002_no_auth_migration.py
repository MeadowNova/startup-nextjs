"""No-auth migration for ResumeForge MVP

Removes User table and simplifies ProcessingJob for anonymous usage
Adds 24-hour auto-cleanup functionality

Revision ID: 002_no_auth
Revises: 001_initial
Create Date: 2024-01-02 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_no_auth'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Migrate to no-auth model"""
    
    # Drop foreign key constraint first
    op.drop_constraint('processing_jobs_user_id_fkey', 'processing_jobs', type_='foreignkey')
    
    # Drop user-related indexes
    op.drop_index('ix_processing_jobs_user_id', table_name='processing_jobs')
    
    # Remove user_id column
    op.drop_column('processing_jobs', 'user_id')
    
    # Add new columns for no-auth model
    op.add_column('processing_jobs', sa.Column('job_id', sa.String(length=36), nullable=False))
    op.add_column('processing_jobs', sa.Column('client_ip', sa.String(length=45), nullable=True))
    op.add_column('processing_jobs', sa.Column('expires_at', sa.DateTime(), nullable=False))
    
    # Create new indexes
    op.create_index('ix_processing_jobs_job_id', 'processing_jobs', ['job_id'], unique=True)
    op.create_index('ix_processing_jobs_expires_at', 'processing_jobs', ['expires_at'])
    
    # Drop the users table entirely
    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_firebase_uid', table_name='users')
    op.drop_index('ix_users_created_at', table_name='users')
    op.drop_index('ix_users_subscription_tier', table_name='users')
    op.drop_table('users')


def downgrade() -> None:
    """Revert to auth model"""
    
    # Recreate users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('firebase_uid', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('subscription_tier', sa.String(length=50), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('profile_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('firebase_uid')
    )
    
    # Recreate user indexes
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_firebase_uid', 'users', ['firebase_uid'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    op.create_index('ix_users_subscription_tier', 'users', ['subscription_tier'])
    
    # Drop no-auth columns and indexes
    op.drop_index('ix_processing_jobs_job_id', table_name='processing_jobs')
    op.drop_index('ix_processing_jobs_expires_at', table_name='processing_jobs')
    
    op.drop_column('processing_jobs', 'expires_at')
    op.drop_column('processing_jobs', 'client_ip')
    op.drop_column('processing_jobs', 'job_id')
    
    # Add back user_id column
    op.add_column('processing_jobs', sa.Column('user_id', sa.Integer(), nullable=False))
    
    # Recreate foreign key and indexes
    op.create_foreign_key('processing_jobs_user_id_fkey', 'processing_jobs', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_index('ix_processing_jobs_user_id', 'processing_jobs', ['user_id'])
