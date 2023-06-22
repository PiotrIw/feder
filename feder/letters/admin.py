from django.contrib import admin
from django.db.models import Q
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from .models import Attachment, Letter, LetterEmailDomain, ReputableLetterEmailTLD


class LetterDirectionListFilter(admin.SimpleListFilter):
    title = _("Letter Direction")  # Displayed in the admin sidebar
    parameter_name = "letter_direction_filter"  # The URL parameter name

    def lookups(self, request, model_admin):
        # Return the filter options as a list of tuples
        return (
            ("outgoing", _("Outgoing")),
            ("incoming", _("Incoming")),
        )

    def queryset(self, request, queryset):
        # Apply the filter to the queryset based on the selected option
        if self.value() == "outgoing":
            return queryset.is_outgoing()
        elif self.value() == "incoming":
            return queryset.is_incoming()


class AttachmentInline(admin.StackedInline):
    """
    Stacked Inline View for Attachment
    """

    model = Attachment


@admin.register(Letter)
class LetterAdmin(admin.ModelAdmin):
    """
    Admin View for Letter
    """

    date_hierarchy = "created"
    list_display = (
        "id",
        "get_record_id",
        "title",
        "get_case",
        "get_monitoring",
        "author",
        "created",
        # "modified",
        "is_draft",
        # "is_incoming",
        "get_outgoing",
        "get_delivery_status",
        "is_spam",
        "email_from",
        "email_to",
        "eml",
        "message_id_header",
    )
    list_filter = (
        "is_spam",
        LetterDirectionListFilter,
        # "created",
        "record__case__monitoring",
        # "modified",
        # "is_outgoing",
    )
    inlines = [AttachmentInline]
    search_fields = (
        "id",
        "title",
        # "body",
        "record__case__name",
        "eml",
        "message_id_header",
        "email_from",
        "email_to",
    )
    raw_id_fields = ("author_user", "author_institution", "record")
    # list_editable = ("is_spam",)
    ordering = ("-id",)
    actions = [
        "delete_selected",
        "mark_spam",
        "mark_probable_spam",
        "mark_spam_unknown",
        "mark_non_spam",
    ]

    @admin.display(
        description=_("Record id"),
        ordering="record__id",
    )
    def get_record_id(self, obj):
        if obj.record is None:
            return None
        return obj.record.id

    @admin.display(
        description=_("Is outgoing"),
        boolean=True,
    )
    def get_outgoing(self, obj):
        return obj.is_outgoing

    @admin.display(
        description=_("Delivery Status"),
    )
    def get_delivery_status(self, obj):
        return obj.emaillog.status_verbose

    @admin.display(
        description=_("Case name"),
        ordering="record__case",
    )
    def get_case(self, obj):
        return obj.record.case

    @admin.display(
        description=_("Monitoring name"),
        ordering="record__case__monitoring",
    )
    def get_monitoring(self, obj):
        if obj.record.case is not None:
            return obj.record.case.monitoring
        return None

    @admin.action(description=_("Mark selected letters as Spam"))
    def mark_spam(modeladmin, request, queryset):
        queryset.update(
            is_spam=Letter.SPAM.spam,
            mark_spam_by=request.user,
            mark_spam_at=timezone.now(),
        )

    @admin.action(description=_("Mark selected letters as Non Spam"))
    def mark_non_spam(modeladmin, request, queryset):
        queryset.update(is_spam=Letter.SPAM.non_spam)

    @admin.action(description=_("Mark selected letters as Spam Unknown"))
    def mark_spam_unknown(modeladmin, request, queryset):
        queryset.update(is_spam=Letter.SPAM.unknown)

    @admin.action(description=_("Mark selected letters as Probable Spam"))
    def mark_probable_spam(modeladmin, request, queryset):
        queryset.update(is_spam=Letter.SPAM.probable_spam)

    # def get_queryset(self, *args, **kwargs):
    #     qs = super().get_queryset(*args, **kwargs)
    #     return qs.with_author()


class ReputableTLDListFilter(admin.SimpleListFilter):
    title = "TLD"
    parameter_name = "tld"

    def lookups(self, request, model_admin):
        return [
            ("reputable", _("Reputable TLDs")),
            ("non_reputable", _("Non-reputable TLDs")),
        ]

    def queryset(self, request, queryset):
        tlds = ReputableLetterEmailTLD.objects.values_list("name", flat=True)
        q_object = Q()
        for tld in tlds:
            q_object |= Q(domain_name__iendswith=tld)
        if self.value() == "reputable":
            return queryset.filter(q_object)
        elif self.value() == "non_reputable":
            return queryset.exclude(q_object)


@admin.register(LetterEmailDomain)
class LetterEmailDomainAdmin(admin.ModelAdmin):
    """
    Admin View for LetterEmailDomain
    """

    list_display = (
        "domain_name",
        "is_trusted_domain",
        "is_monitoring_email_to_domain",
        "is_non_spammer_domain",
        "is_spammer_domain",
        "email_to_count",
        "email_from_count",
    )
    list_filter = (
        "is_trusted_domain",
        "is_monitoring_email_to_domain",
        "is_non_spammer_domain",
        "is_spammer_domain",
        ReputableTLDListFilter,
    )
    search_fields = ("domain_name",)
    ordering = ("-email_from_count",)
    list_editable = ("is_spammer_domain", "is_non_spammer_domain")


@admin.register(ReputableLetterEmailTLD)
class ReputableLetterEmailTLDAdmin(admin.ModelAdmin):
    """
    Admin View for ReputableLetterEmailTLD
    """

    list_display = ("id", "name")
    search_fields = ("name",)
    ordering = ("name",)
