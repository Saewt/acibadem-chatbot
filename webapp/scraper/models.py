from django.db import models
from pgvector.django import VectorField


class WebPage(models.Model):
    url = models.URLField(unique=True, max_length=500)
    title = models.CharField(max_length=500, blank=True)
    content_text = models.TextField(blank=True)
    raw_html = models.TextField(blank=True)
    scraped_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or self.url


class ContentChunk(models.Model):
    page = models.ForeignKey(WebPage, on_delete=models.CASCADE, related_name='chunks')
    text = models.TextField()
    embedding = VectorField(dimensions=768, null=True, blank=True)
    chunk_index = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.page}"

    class Meta:
        ordering = ['page', 'chunk_index']
