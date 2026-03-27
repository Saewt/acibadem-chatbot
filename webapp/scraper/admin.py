from django.contrib import admin
from .models import WebPage, ContentChunk


class ContentChunkInline(admin.TabularInline):
    model = ContentChunk
    extra = 0
    fields = ('chunk_index', 'short_text', 'has_embedding')
    readonly_fields = ('chunk_index', 'short_text', 'has_embedding')

    def short_text(self, obj):
        return obj.text[:100]
    short_text.short_description = 'Text'

    def has_embedding(self, obj):
        return obj.embedding is not None
    has_embedding.boolean = True
    has_embedding.short_description = 'Embedded'


@admin.register(WebPage)
class WebPageAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'url', 'chunk_count', 'scraped_at')
    search_fields = ('title', 'url')
    list_filter = ('scraped_at',)
    inlines = [ContentChunkInline]

    def chunk_count(self, obj):
        return obj.chunks.count()
    chunk_count.short_description = 'Chunks'


@admin.register(ContentChunk)
class ContentChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'short_text', 'page', 'chunk_index', 'has_embedding')
    list_filter = ('page',)
    search_fields = ('text',)
    raw_id_fields = ('page',)

    def short_text(self, obj):
        return obj.text[:100]
    short_text.short_description = 'Text'

    def has_embedding(self, obj):
        return obj.embedding is not None
    has_embedding.boolean = True
    has_embedding.short_description = 'Embedded'
