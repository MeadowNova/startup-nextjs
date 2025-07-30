"""Initial migration for ResumeForge

Creates User and ProcessingJob tables with proper indexes and constraints
Compatible with any PostgreSQL provider (Vercel Postgres, Supabase, etc.)

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema"""
    
    # Create users table
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
    
    # Create indexes for users table
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_firebase_uid', 'users', ['firebase_uid'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    op.create_index('ix_users_subscription_tier', 'users', ['subscription_tier'])
    
    # Create processing_jobs table
    op.create_table(
        'processing_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        
        # Input files and data
        sa.Column('original_resume_blob_url', sa.String(length=500), nullable=True),
        sa.Column('job_description', sa.Text(), nullable=True),
        sa.Column('job_description_file_url', sa.String(length=500), nullable=True),
        sa.Column('job_title', sa.String(length=200), nullable=True),
        sa.Column('company_name', sa.String(length=200), nullable=True),
        
        # Processing Results
        sa.Column('optimized_resume_blob_url', sa.String(length=500), nullable=True),
        sa.Column('cover_letter_blob_url', sa.String(length=500), nullable=True),
        sa.Column('package_zip_blob_url', sa.String(length=500), nullable=True),
        
        # AI Processing Metadata
        sa.Column('resume_analysis', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('ats_keyword_map', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('change_log', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('layout_coordinates', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        
        # Performance Metrics
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('ai_tokens_used', sa.Integer(), nullable=True),
        sa.Column('openai_cost_usd', sa.Float(), nullable=True),
        
        # Error Handling
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_code', sa.String(length=50), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True, default=0),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        # Additional metadata
        sa.Column('processing_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('user_feedback', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create indexes for processing_jobs table
    op.create_index('ix_processing_jobs_user_id', 'processing_jobs', ['user_id'])
    op.create_index('ix_processing_jobs_status', 'processing_jobs', ['status'])
    op.create_index('ix_processing_jobs_created_at', 'processing_jobs', ['created_at'])
    op.create_index('ix_processing_jobs_completed_at', 'processing_jobs', ['completed_at'])
    op.create_index('ix_processing_jobs_company_name', 'processing_jobs', ['company_name'])
    
    # Composite indexes for common queries
    op.create_index(
        'ix_processing_jobs_user_status', 
        'processing_jobs', 
        ['user_id', 'status']
    )
    op.create_index(
        'ix_processing_jobs_user_created', 
        'processing_jobs', 
        ['user_id', 'created_at']
    )
    
    # Create job_history table for tracking user activity
    op.create_table(
        'job_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('processing_job_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['processing_job_id'], ['processing_jobs.id'], ondelete='SET NULL')
    )
    
    # Create indexes for job_history table
    op.create_index('ix_job_history_user_id', 'job_history', ['user_id'])
    op.create_index('ix_job_history_created_at', 'job_history', ['created_at'])
    op.create_index('ix_job_history_action', 'job_history', ['action'])
    
    # Create user_subscriptions table for subscription management
    op.create_table(
        'user_subscriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('subscription_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(), nullable=True),
        sa.Column('stripe_subscription_id', sa.String(length=100), nullable=True),
        sa.Column('stripe_customer_id', sa.String(length=100), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('stripe_subscription_id')
    )
    
    # Create indexes for user_subscriptions table
    op.create_index('ix_user_subscriptions_user_id', 'user_subscriptions', ['user_id'])
    op.create_index('ix_user_subscriptions_status', 'user_subscriptions', ['status'])
    op.create_index('ix_user_subscriptions_expires_at', 'user_subscriptions', ['expires_at'])
    
    # Create usage_tracking table for monitoring API usage
    op.create_table(
        'usage_tracking',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('processing_job_id', sa.Integer(), nullable=True),
        sa.Column('resource_type', sa.String(length=50), nullable=False),  # 'ai_tokens', 'pdf_processing', etc.
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['processing_job_id'], ['processing_jobs.id'], ondelete='SET NULL')
    )
    
    # Create indexes for usage_tracking table
    op.create_index('ix_usage_tracking_user_id', 'usage_tracking', ['user_id'])
    op.create_index('ix_usage_tracking_created_at', 'usage_tracking', ['created_at'])
    op.create_index('ix_usage_tracking_resource_type', 'usage_tracking', ['resource_type'])
    
    # Composite index for usage queries
    op.create_index(
        'ix_usage_tracking_user_resource_date', 
        'usage_tracking', 
        ['user_id', 'resource_type', 'created_at']
    )


def downgrade() -> None:
    """Drop all tables and indexes"""
    
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('usage_tracking')
    op.drop_table('user_subscriptions')
    op.drop_table('job_history')
    op.drop_table('processing_jobs')
    op.drop_table('users')
