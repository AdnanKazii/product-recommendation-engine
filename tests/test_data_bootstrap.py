from src.recsys.data_prep import REQUIRED_FILES

from app.data_bootstrap import ensure_data_available


def test_skips_download_when_all_files_already_present(tmp_path, monkeypatch):
    for filename in REQUIRED_FILES.values():
        (tmp_path / filename).touch()

    called = False

    def fake_download(data_dir):
        nonlocal called
        called = True

    monkeypatch.setattr("app.data_bootstrap._download_from_kaggle", fake_download)

    ensure_data_available(tmp_path)

    assert called is False


def test_downloads_when_files_are_missing(tmp_path, monkeypatch):
    called_with = []
    monkeypatch.setattr(
        "app.data_bootstrap._download_from_kaggle", lambda data_dir: called_with.append(data_dir)
    )

    ensure_data_available(tmp_path)

    assert called_with == [tmp_path]
