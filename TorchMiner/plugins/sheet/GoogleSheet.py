# -*- coding:utf-8 -*-
import httplib2

from TorchMiner.plugins.sheet import Sheet, _async

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


def num_to_letter(num):
    num += 1
    letters = ""
    while num:
        mod = (num - 1) % 26
        letters += chr(mod + 65)
        num = (num - 1) // 26
    return "".join(reversed(letters))


class GoogleSheet(Sheet):
    def _create_experiment_row(self):
        pass

    def __init__(
            self, sheet_id, service_account_file, meta_prefix="", build_kwargs=None, proxy=None
    ):
        super().__init__()
        if proxy:  # If proxy is Used to connect to Google Api Server
            # Proxy Should be a dict :{"ip":"ip_address","port":"your port"}
            self.http = httplib2.Http(proxy_info=httplib2.ProxyInfo(
                httplib2.socks.PROXY_TYPE_HTTP, proxy["ip"], proxy["port"]
            ))
        else:
            self.http = None

        if build_kwargs is None:
            build_kwargs = {}

        self.sheet_id = sheet_id
        self.meta_prefix = meta_prefix

        credentials = service_account.Credentials.from_service_account_file(
            service_account_file, http=self.http
        )
        service = build("sheets", "v4", credentials=credentials, **build_kwargs, http=self.http)
        self.drive = build("drive", "v3", credentials=credentials, http=self.http)
        self.sheet = service.spreadsheets()
        self.drive_folder_id = self._prepare_drive_directory()

    def _meta(self, key):
        return f"{self.meta_prefix}{key}"

    def _index_of(self, key):
        search = {
            "dataFilters": [
                {"developerMetadataLookup": {"metadataKey": self._meta(key)}}
            ]
        }
        result = (
            self.sheet.developerMetadata()
                .search(spreadsheetId=self.sheet_id, body=search)
                .execute()
        )
        if len(result.items()) == 0:
            return False
        else:
            return result["matchedDeveloperMetadata"][0]["developerMetadata"]["location"]["dimensionRange"][
                "startIndex"]

    _exists = _index_of

    @_async
    def reset_index(self):
        self.banner_index = self._create_banner_dimension()
        self.title_index = self._create_title_dimension()
        self.endcol_index = self._create_end_column_divider()
        self.experiment_row_index = self._insert_dimension(
            self.experiment_row_name, self.title_index + 1, "ROWS"
        )

    @_async
    def prepare(self):
        self.reset_index()
        self.update("code", self.code)

    @property
    def dark_bg(self):
        return {
            "red": 0.10980392156862745,
            "green": 0.5686274509803921,
            "blue": 0.6039215686274509,
        }

    @property
    def white(self):
        return {"red": 1.0, "green": 1.0, "blue": 1.0}

    @property
    def light_bg(self):
        return {
            "red": 0.9411764705882353,
            "green": 1.0,
            "blue": 0.9882352941176471,
        }

    def _create_banner_dimension(self):
        return self._insert_dimension(self.banner_row_name, 0, "ROWS")

    def _create_title_dimension(self):
        return self._insert_dimension(self.title_row_name, 1, "ROWS")

    def _create_end_column_divider(self):
        icol = self._insert_dimension(self.end_column_name, 0, "COLUMNS")
        requests = [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": 0,
                        "startColumnIndex": icol,
                        "endColumnIndex": icol + 1,
                        "startRowIndex": self.title_index,
                        "endRowIndex": self.title_index + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": self.dark_bg,
                            "textFormat": {
                                "foregroundColor": self.white,
                                "bold": False,
                                "fontSize": 12,
                            },
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE",
                        }
                    },
                    "fields": "*",
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": 0,
                        "dimension": "COLUMNS",
                        "startIndex": icol,
                        "endIndex": icol + 1,
                    },
                    "properties": {
                        "hiddenByUser": True,
                    },
                    "fields": "hiddenByUser",
                }
            },
        ]

        self.sheet.batchUpdate(
            spreadsheetId=self.sheet_id, body={"requests": requests}
        ).execute()
        return icol

    def _insert_dimension(self, row_name, index, dim, extra_request=None):
        result = self._exists(row_name)
        if result is not False:
            return result

        if extra_request is None:
            extra_request = []

        create_row_request = {
            "insertDimension": {
                "range": {
                    "sheetId": 0,
                    "dimension": dim,
                    "startIndex": index,
                    "endIndex": index + 1,
                },
                "inheritFromBefore": False,
            }
        }

        assign_name_request = {
            "createDeveloperMetadata": {
                "developerMetadata": {
                    "metadataKey": self._meta(row_name),
                    "metadataValue": self._meta(row_name),
                    "location": {
                        "dimensionRange": {
                            "sheetId": 0,
                            "dimension": dim,
                            "startIndex": index,
                            "endIndex": index + 1,
                        }
                    },
                    "visibility": "DOCUMENT",
                }
            }
        }

        body = {
            "requests": [create_row_request, assign_name_request, *extra_request],
        }

        self.sheet.batchUpdate(spreadsheetId=self.sheet_id, body=body).execute()
        return self._exists(row_name)

    @_async
    def onready(self):
        banner = """
        TorchMiner Official Google Sheet Plugin.
        If you found it's useful, please considering star the project at https://github.com/InEase/TorchMiner.
        """
        icol_end = self._index_of(self.end_column_name)
        icol_start = self._index_of(self.end_column_name) - len(self.columns)
        merge_cells = {
            "mergeCells": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": self.banner_index,
                    "endRowIndex": self.banner_index + 1,
                    "startColumnIndex": icol_start,
                    "endColumnIndex": icol_end,
                },
                "mergeType": "MERGE_ALL",
            }
        }

        change_cell = {
            "repeatCell": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": self.banner_index,
                    "endRowIndex": self.banner_index + 1,
                    "startColumnIndex": icol_start,
                    "endColumnIndex": icol_end,
                },
                "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                "fields": "*",
            }
        }

        auto_resize = {
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": 0,
                    "dimension": "ROWS",
                    "startIndex": self.banner_index,
                    "endIndex": self.banner_index + 1,
                }
            }
        }

        body = {"requests": [merge_cells, change_cell, auto_resize]}

        self.sheet.batchUpdate(spreadsheetId=self.sheet_id, body=body).execute()
        self._update_cells(
            f"{self._num_to_letter(icol_start)}{self.banner_index + 1}", [banner]
        )

    @_async
    def create_column(self, key, title, size=None):
        super().create_column(key, title)
        col_index = self._index_of(self.end_column_name)

        change_cell_request = {
            "repeatCell": {
                "range": {
                    "sheetId": 0,
                    "startColumnIndex": col_index,
                    "endColumnIndex": col_index + 1,
                    "startRowIndex": self.experiment_row_index,
                    "endRowIndex": self.experiment_row_index + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": self.light_bg,
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }
                },
                "fields": "*",
            },
        }

        assign_name_request = {
            "createDeveloperMetadata": {
                "developerMetadata": {
                    "metadataKey": self._meta("__minetorch_column__"),
                    "metadataValue": key,
                    "location": {
                        "dimensionRange": {
                            "sheetId": 0,
                            "dimension": "COLUMNS",
                            "startIndex": col_index,
                            "endIndex": col_index + 1,
                        }
                    },
                    "visibility": "DOCUMENT",
                }
            }
        }

        index = self._insert_dimension(
            key, col_index, "COLUMNS", [change_cell_request, assign_name_request]
        )
        self._update_cells(
            f"{self._num_to_letter(index)}{self.title_index + 1}", [title]
        )

    def _update_cells(self, a1, values):
        value_range = {"range": a1, "majorDimension": "ROWS", "values": [values]}
        try:
            self.sheet.values().update(
                spreadsheetId=self.sheet_id,
                range=a1,
                valueInputOption="USER_ENTERED",
                body=value_range,
            ).execute()
        except Exception as e:
            self.logger.warn(f"Update sheet failed with {e}")
            return

    @_async
    def flush(self):
        irow = self._index_of(self.experiment_row_name)
        column_indices = self._get_column_indices()
        for key, value in self.cached_row_data.items():
            raw_value = value.get("raw")
            processor = value.get("processor")
            if processor is None:
                value = raw_value
            else:
                value = getattr(self, f"_process_{processor}")(key, raw_value)
            icol = column_indices[key]
            self._update_cells(f"{self._num_to_letter(icol)}{irow + 1}", [value])
        self.cached_row_data = {}

    def _process_upload_image(self, key, value, retry=True):
        try:
            image_id = self._upload_drive_image(key, value)
            return f'=IMAGE("https://drive.google.com/uc?export=view&id={image_id}", 2)'
        except HttpError as e:
            if retry:
                return self._process_upload_image(key, value)
            raise e

    def _process_repr(self, key, value):
        return repr(value)

    def _get_column_indices(self):
        search = {
            "dataFilters": [
                {
                    "developerMetadataLookup": {
                        "metadataKey": self._meta("__minetorch_column__")
                    }
                }
            ]
        }
        r = (
            self.sheet.developerMetadata()
                .search(spreadsheetId=self.sheet_id, body=search)
                .execute()
        )
        result = {}
        for item in r["matchedDeveloperMetadata"]:
            column_key = item["developerMetadata"]["metadataValue"]
            index = item["developerMetadata"]["location"]["dimensionRange"][
                "startIndex"
            ]
            result[column_key] = index
        return result

    def _upload_drive_image(self, key, value, retry=True):
        try:
            file_metadata = {"name": key, "parents": [self.drive_folder_id]}
            media = MediaFileUpload(value, mimetype="image/png")
            file = (
                self.drive.files()
                    .create(body=file_metadata, media_body=media, fields="id")
                    .execute()
            )
            return file.get("id")
        except HttpError as e:
            if not retry:
                raise e
            self.drive_folder_id = self._prepare_drive_directory()
            self._upload_drive_image(key, value, retry=False)

    def _prepare_drive_directory(self):
        try:
            result = (
                self.drive.files()
                    .list(q="name='TorchMiner_assets'", fields="files(id)")
                    .execute()
            )
            dir_id = result["files"][0]["id"]
        except IndexError:
            file_metadata = {
                "name": "TorchMiner_assets",
                "mimeType": "application/vnd.google-apps.folder",
            }
            file = self.drive.files().create(body=file_metadata, fields="id").execute()
            dir_id = file.get("id")

        self.drive.permissions().create(
            fileId=dir_id, body={"role": "writer", "type": "anyone"}
        ).execute()
        return dir_id
