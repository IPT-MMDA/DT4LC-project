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

- **Source:** https://huggingface.co/MykolaL/DelineateAnything
- **License:** MIT
- **Copyright:** Mykola Lavreniuk
- **Description:** Agricultural field boundary detection model based on SAM (Segment Anything Model).
- **Paper:** https://arxiv.org/abs/2401.05923
- **Citation:**
  ```
  @article{lavreniuk2024delineate,
    title={Delineate-Anything: Segment Anything Model for Agricultural Field Boundary Delineation},
    author={Lavreniuk, Mykola and others},
    journal={arXiv preprint arXiv:2401.05923},
    year={2024}
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

### MIT License (Delineate-Anything)

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Model Download

Models are downloaded on-demand when users request features that require them. The models are cached locally in:

- **Local development:** `~/.cache/dt4lc/models/`
- **Docker:** `/app/models/` (baked into image at build time)

Use the Models page in the UI to manage model downloads and deletions.
