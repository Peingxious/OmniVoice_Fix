# Copyright    2026  OmniVoice Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for decoupled FastAPI Sentence Studio API."""

from starlette.testclient import TestClient
from omnivoice.server.app import create_app


def test_api_static_index():
    """Verify that root URL serves web/index.html."""
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert "OmniVoice 分句配音工作室" in res.text
    assert "Vue" in res.text or "createApp" in res.text


def test_api_voices_endpoint():
    """Verify that /api/voices returns list and default role."""
    client = TestClient(create_app())
    res = client.get("/api/voices")
    assert res.status_code == 200
    data = res.json()
    assert "voices" in data
    assert isinstance(data["voices"], list)
    if data["voices"]:
        assert "filename" in data["voices"][0]
        assert "size_kb" in data["voices"][0]


def test_api_split_endpoint():
    """Verify that /api/split correctly splits multi-line text."""
    client = TestClient(create_app())
    payload = {
        "text": "\n第一句话，连句不拆。\n\n第二句话！感叹号保留。\n第三句话？问句保留。\n"
    }
    res = client.post("/api/split", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 3
    assert data["sentences"][0] == "第一句话，连句不拆。"
    assert data["sentences"][1] == "第二句话！感叹号保留。"
    assert data["sentences"][2] == "第三句话？问句保留。"


if __name__ == "__main__":
    test_api_static_index()
    test_api_voices_endpoint()
    test_api_split_endpoint()
    print("All server API tests passed successfully!")
