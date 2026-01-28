import os
import tempfile
import pandas as pd
import pytest

from semantica.ingest.pandas_ingestor import PandasIngestor


def write_temp_csv(content: str, encoding="utf-8"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    with open(tmp.name, "w", encoding=encoding) as f:
        f.write(content)
    return tmp.name


# -------------------------------------------------------
# Test 1: Encoding Detection (international entities)
# -------------------------------------------------------

def test_from_csv_detects_encoding():
    # Latin-1 encoded content
    content = (
        "entity_name,location\n"
        "Siemens,München\n"
        "Telefónica,España\n"
    )

    path = write_temp_csv(content, encoding="latin-1")

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert data.columns == ["entity_name", "location"]
    assert data.dataframe.iloc[0]["entity_name"] == "Siemens"

    os.remove(path)


def test_from_csv_detects_tab_delimiter():
    content = (
        "user_id\trole\n"
        "1\tadmin\n"
        "2\tuser\n"
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert data.columns == ["user_id", "role"]
    assert data.dataframe.iloc[0]["role"] == "admin"

    os.remove(path)


def test_from_csv_handles_quoted_fields_with_commas():
    content = (
        "company,revenue\n"
        '"Acme, Inc.",100\n'
        '"Widgets, LLC",200\n'
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert data.columns == ["company", "revenue"]
    assert data.dataframe.iloc[0]["company"] == "Acme, Inc."
    assert int(data.dataframe.iloc[1]["revenue"]) == 200

    os.remove(path)


def test_from_csv_handles_multiline_quoted_fields():
    # Embed actual newlines within quoted fields
    content = "id,notes\n1,\"line1\nline2\"\n2,\"alpha\nbeta\"\n"

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert "\n" in data.dataframe.iloc[0]["notes"]
    assert data.dataframe.iloc[1]["notes"].split("\n")[1] == "beta"

    os.remove(path)


def test_from_csv_no_header_override():
    content = (
        "colA,colB\n"
        "x,1\n"
        "y,2\n"
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path, header=None)

    assert data.row_count == 3
    assert data.columns == [0, 1]
    assert list(data.dataframe.iloc[0]) == ["colA", "colB"]

    os.remove(path)


def test_from_csv_with_chunksize_concatenates():
    rows = ["a,b", "1,x", "2,y", "3,z", "4,w"]
    content = "\n".join(rows) + "\n"

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path, chunksize=2)

    assert data.row_count == 4
    assert data.metadata.get("chunksize") == 2
    assert list(data.dataframe["a"]) == [1, 2, 3, 4]

    os.remove(path)


def test_from_csv_preserves_nan_values():
    content = (
        "name,score\n"
        "alice,\n"
        "bob,10\n"
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert pd.isna(data.dataframe.iloc[0]["score"]) is True
    assert int(data.dataframe.iloc[1]["score"]) == 10

    os.remove(path)


# -------------------------------------------------------
# Test 2: Delimiter Detection (semicolon separated)
# -------------------------------------------------------

def test_from_csv_detects_delimiter():
    content = (
        "company;sector\n"
        "CrowdStrike;Cybersecurity\n"
        "Fortinet;Network Security\n"
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    assert data.row_count == 2
    assert data.columns == ["company", "sector"]
    assert data.dataframe.iloc[1]["company"] == "Fortinet"

    os.remove(path)


# -------------------------------------------------------
# Test 3: Bad Rows Are Skipped (malformed threat feed)
# -------------------------------------------------------

def test_from_csv_skips_bad_rows():
    content = (
        "indicator,type\n"
        "192.168.1.10,IP\n"
        "this,is,too,many,columns\n"   # malformed row
        "evil-domain.com,Domain\n"
    )

    path = write_temp_csv(content)

    ingestor = PandasIngestor()
    data = ingestor.from_csv(path)

    # Malformed row should be skipped
    assert data.row_count == 2
    assert list(data.dataframe["indicator"]) == [
        "192.168.1.10",
        "evil-domain.com",
    ]

    os.remove(path)
