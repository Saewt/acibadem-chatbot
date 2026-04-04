from django.db import models
from pgvector.django import HnswIndex, VectorField


class WebPage(models.Model):
    SOURCE_CHOICES = [
        ('main_site', 'Main Site'),
        ('bologna', 'Bologna'),
        ('structured', 'Structured'),
    ]

    url = models.URLField(unique=True, max_length=500)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='main_site')
    language = models.CharField(max_length=10, default='tr')
    title = models.CharField(max_length=500, blank=True)
    content_text = models.TextField(blank=True)
    raw_html = models.TextField(blank=True)
    content_hash = models.CharField(max_length=64, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    scraped_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_seen_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or self.url

    class Meta:
        ordering = ['title', 'url']


class ContentChunk(models.Model):
    page = models.ForeignKey(WebPage, on_delete=models.CASCADE, related_name='chunks')
    text = models.TextField()
    embedding = VectorField(dimensions=384, null=True, blank=True)
    chunk_index = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.page}"

    class Meta:
        ordering = ['page', 'chunk_index']
        indexes = [
            HnswIndex(
                fields=['embedding'],
                name='contentchunk_embedding_hnsw_idx',
                opclasses=['vector_cosine_ops'],
            )
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['page', 'chunk_index'], name='unique_chunk_per_page_position'
            )
        ]
