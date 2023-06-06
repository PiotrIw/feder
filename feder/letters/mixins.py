from django.utils.safestring import mark_safe
from guardian.shortcuts import get_anonymous_user

from .models import Letter


class LetterObjectFeedMixin:
    """A mixins to view (feed) to provide easy way to generate
    feed of letters related to selected object

    Attributes:
        filter_field (str): path to related object to Letter
        kwargs_name (str): name used in urlpatterns
        model (model): model used to select object related to Letter
    """

    model = None
    filter_field = None
    kwargs_name = None

    def get_object(self, request, **kwargs):
        return self.model.objects.get(pk=kwargs.get(self.kwargs_name))

    def link(self, obj):
        return obj.get_absolute_url()

    def get_items(self, obj):
        return (
            Letter.objects.with_feed_items()
            .filter(**{self.filter_field: obj})
            .exclude_spam()
            .for_user(get_anonymous_user())
            .order_by("-created")[:30]
        )


class LetterSummaryTableMixin:
    def render_summary_table(self):
        """
        returns a summary html table with letter count for each filetering option
        in LetterFilter created filter field
        """
        qs = Letter.objects.all()
        table_html = """
        <table class="table table-bordered compact" style="width: 100%">
        <thead><tr>
        <th colspan="2"> </th>
        <th colspan="2">Nierozpoznane</th>
        <th colspan="2">Rozpoznane</th>
        </tr></thead>        <thead><tr>
        <th>Okres</th>
        <th>Liczba listów</th>
        <th style="white-space: nowrap;">Nie spam</th>
        <th>Spam</th>
        <th>Spam</th>
        <th style="white-space: nowrap;">Nie spam</th>
        </tr></thead>"""
        choices = self.filterset.filters["created"].choices
        filters = self.filterset.filters["created"].filters
        for choice in choices:
            filtered_qs = filters[choice[0]](qs, "created")
            letters_count = filtered_qs.count()
            assigned_qs = filtered_qs.filter(record__case__isnull=False)
            unassigned_qs = filtered_qs.filter(record__case__isnull=True)
            unassigned_nonspam_count = unassigned_qs.exclude(
                is_spam=Letter.SPAM.spam
            ).count()
            unassigned_spam_count = unassigned_qs.filter(
                is_spam=Letter.SPAM.spam
            ).count()
            assigned_spam_count = assigned_qs.filter(is_spam=Letter.SPAM.spam).count()
            assigned_nonspam_count = assigned_qs.exclude(
                is_spam=Letter.SPAM.spam
            ).count()
            table_html += f"""<tr>
            <td>{choice[1]}</td>
            <td style="text-align: right;">{letters_count}</td>
            <td style="text-align: right;">{unassigned_nonspam_count}</td>
            <td style="text-align: right;">{unassigned_spam_count}</td>
            <td style="text-align: right;">{assigned_spam_count}</td>
            <td style="text-align: right;">{assigned_nonspam_count}</td>
            </tr>"""
            print(
                choice[1],
                letters_count,
                (
                    letters_count
                    - unassigned_nonspam_count
                    - unassigned_spam_count
                    - assigned_nonspam_count
                    - assigned_spam_count
                ),
            )
        table_html += "</table>"
        return mark_safe(table_html)
