#!/usr/bin/env python3
"""
CodeWiki Frontend Module

Web interface components for the documentation generation service.
"""

from .background_worker import BackgroundWorker
from .cache_manager import CacheManager
from .github_processor import GitHubRepoProcessor
from .models import CacheEntry, JobStatus, JobStatusResponse, RepositorySubmission
from .routes import WebRoutes
from .web_app import app, main

__all__ = [
    "app",
    "main",
    "JobStatus",
    "JobStatusResponse",
    "RepositorySubmission",
    "CacheEntry",
    "CacheManager",
    "BackgroundWorker",
    "GitHubRepoProcessor",
    "WebRoutes",
]
