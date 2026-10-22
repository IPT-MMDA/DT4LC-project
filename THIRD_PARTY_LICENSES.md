# Third-Party Model Licenses

This document lists the third-party machine learning models used in DT4LC and their respective licenses.

## Prithvi EO v1 (100M)

- **Source:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- **License:** Apache License 2.0
- **Copyright:** IBM and NASA
- **Description:** NASA/IBM geospatial foundation model for Earth observation. Extracts temporal features and embeddings from HLS satellite data.
- **Citation:**
  ```
  @misc{prithvi-eo-1.0,
    author = {IBM and NASA},
    title = {Prithvi EO 1.0},
    year = {2023},
    url = {https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M}
  }
  ```

## Delineate-Anything

- **Source:** https://huggingface.co/MykolaL/DelineateAnything (weights `DelineateAnything-S.pt`) and https://github.com/Lavreniuk/Delineate-Anything (inference code, branch `main`), both downloaded by the model manager
- **License:** GNU Affero General Public License v3.0 (AGPL-3.0): `license: agpl-3.0` on the Hugging Face model card, and the AGPL-3.0 `LICENSE` file of the GitHub repository
- **Authors:** Mykola Lavreniuk, Nataliia Kussul, Andrii Shelestov, Bohdan Yailymov, Yevhenii Salii, Volodymyr Kuzin, Zoltan Szantoi
- **Description:** Instance segmentation model for agricultural field boundary delineation on satellite imagery.
- **Paper:** https://arxiv.org/abs/2504.02534
- **Citation:**
  ```
  @misc{lavreniuk2025delineate,
    title={Delineate Anything: Resolution-Agnostic Field Boundary Delineation on Satellite Imagery},
    author={Lavreniuk, Mykola and Kussul, Nataliia and Shelestov, Andrii and Yailymov, Bohdan and Salii, Yevhenii and Kuzin, Volodymyr and Szantoi, Zoltan},
    year={2025},
    eprint={2504.02534},
    archivePrefix={arXiv},
    doi={10.48550/arXiv.2504.02534}
  }
  ```

---

## License Notices

### Apache License 2.0 (Prithvi)

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### GNU AGPL-3.0 (Delineate-Anything)

Delineate-Anything is distributed under the GNU Affero General Public License, version 3, the same licence as DT4LC. The full text is in this repository's [LICENSE](LICENSE) file and at https://www.gnu.org/licenses/agpl-3.0.html.

---

## Model Download

Models are not bundled with DT4LC; they are downloaded on request (see "Model weights" in the [README](README.md#model-weights)) and cached in:

- **Local development:** `resources/.cache/models/<model_id>/` (override with the `DT4LC_MODEL_CACHE` environment variable)
- **Docker:** `/app/resources/.cache/models/<model_id>/` (`DT4LC_MODEL_CACHE` in `docker-compose.yml`), inside the mounted `./resources` directory; downloaded at runtime, not part of the image

Use the Models page in the UI to manage model downloads and deletions.
