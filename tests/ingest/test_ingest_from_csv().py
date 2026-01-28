import os
import tempfile
import pandas as pd
import pytest

from semantica.ingest.pandas_ingestor import PandasIngestor


# -------------------------------------------------------
# Helper to write temporary CSV
# -------------------------------------------------------

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
