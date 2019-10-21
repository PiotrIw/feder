import django_filters
from braces.views import LoginRequiredMixin
from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import EmptyPage, Paginator
from django.utils import six
from django.views.generic.detail import BaseDetailView
from guardian.mixins import PermissionRequiredMixin
from guardian.shortcuts import assign_perm
from sendfile import sendfile


class ExtraListMixin(object):
    """Mixins for view to add additional paginated object list

    Attributes:
        extra_list_context (str): Name of extra list context
        paginate_by (int): Number of added objects per page
    """

    paginate_by = 25
    extra_list_context = "object_list"

    def paginator(self, object_list):
        """A Method to paginate object_list accordingly.

        Args:
            object_list (QuerySet): A list of object to paginate

        Returns:
            Page: A page for current requests
        """
        paginator = Paginator(object_list, self.paginate_by)
        try:
            return paginator.page(self.kwargs.get("page", 1))
        except EmptyPage:
            # If page is out of range (e.g. 9999), deliver last page of results.
            return paginator.page(paginator.num_pages)

    def get_object_list(self, obj):
        """A method to return object list to additional list. This should be overriden.

        Args:
            obj: The object the view is displaying.

        Returns:
            QuerySet: A list of object to paginated
        Raises:
            ImproperlyConfigured: The method was not overrided.
        """
        raise ImproperlyConfigured(
            "{0} is missing a permissions to assign. Define {0}.permission "
            "or override {0}.get_permission().".format(self.__class__.__name__)
        )

    def get_context_data(self, **kwargs):
        context = super(ExtraListMixin, self).get_context_data(**kwargs)
        object_list = self.get_object_list(self.object)
        context[self.extra_list_context] = self.paginator(object_list)
        return context


class RaisePermissionRequiredMixin(LoginRequiredMixin, PermissionRequiredMixin):
    """Mixin to verify object permission with preserve correct status code in view
    """

    raise_exception = True
    redirect_unauthenticated_users = True


class AttrPermissionRequiredMixin(RaisePermissionRequiredMixin):
    """Mixin to verify object permission in SingleObjectView

    Attributes:
        permission_attribute (str): A path to traverse from object to permission object
    """

    permission_attribute = None

    @staticmethod
    def _resolve_path(obj, path=None):
        """Resolve django-like path eg. object2__object3 for object

        Args:
            obj: The object the view is displaying.
            path (str, optional): Description

        Returns:
            A oject at end of resolved path
        """
        if path:
            for attr_name in path.split("__"):
                obj = getattr(obj, attr_name)
        return obj

    def get_permission_object(self):
        obj = super(AttrPermissionRequiredMixin, self).get_object()
        return self._resolve_path(obj, self.permission_attribute)

    def get_object(self):
        if not hasattr(self, "object"):
            self.object = super(AttrPermissionRequiredMixin, self).get_object()
        return self.object


class AutocompletePerformanceMixin(object):
    """A mixin to improve autocomplete to limit SELECTed fields

    Attributes:
        select_only (list): List of fields to select
    """

    select_only = None

    def choices_for_request(self, *args, **kwargs):
        qs = super(AutocompletePerformanceMixin, self).choices_for_request(
            *args, **kwargs
        )
        if self.select_only:
            qs = qs.only(*self.select_only)
        return qs


class DisabledWhenFilterSetMixin(django_filters.filterset.BaseFilterSet):
    @property
    def qs(self):
        if not hasattr(self, "_qs") and self.is_bound and self.form.is_valid():
            for name in list(self.filters.keys()):
                filter_ = self.filters[name]
                value = self.form.cleaned_data.get(name)
                enabled_test = getattr(
                    filter_, "check_enabled", lambda _: True
                )  # legacy-filter compatible
                if not enabled_test(self.form.cleaned_data):
                    del self.filters[name]
        return super(DisabledWhenFilterSetMixin, self).qs


class DisabledWhenFilterMixin(object):
    def __init__(self, *args, **kwargs):
        self.disabled_when = kwargs.pop("disabled_when", [])
        super(DisabledWhenFilterMixin, self).__init__(*args, **kwargs)

    def check_enabled(self, form_data):
        return not any(form_data[field] for field in self.disabled_when)


class BaseXSendFileView(BaseDetailView):
    file_field = None
    send_as_attachment = None

    def get_file_field(self):
        return self.file_field

    def get_file_path(self, object):
        return getattr(object, self.get_file_field()).path

    def get_sendfile_kwargs(self, context):
        return dict(
            request=self.request,
            filename=self.get_file_path(context["object"]),
            attachment=self.send_as_attachment,
        )

    def render_to_response(self, context):
        return sendfile(**self.get_sendfile_kwargs(context))
