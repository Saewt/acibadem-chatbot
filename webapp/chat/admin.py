from django.contrib import admin
from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ('role', 'content', 'created_at')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'created_at', 'message_count')
    search_fields = ('title',)
    inlines = [MessageInline]

    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'role', 'short_content', 'conversation', 'created_at')
    list_filter = ('role',)
    search_fields = ('content',)

    def short_content(self, obj):
        return obj.content[:80]
    short_content.short_description = 'Content'
