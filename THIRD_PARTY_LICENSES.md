# Third-Party Licenses

This document lists the third-party machine learning models and fonts used in DT4LC with their licenses, the basemap and imagery credits the interface shows, and the source and license of the oblast boundaries on the map.

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

Operators pre-download models in Settings → Models (`/app/settings`), where they also choose who may download a model on first use (`MODEL_DOWNLOAD_ON_DEMAND`). Otherwise a model that is not installed is downloaded the first time an analysis needs it, if the caller's role may download. The Capabilities page shows each model's state. Deleting a model is an operator's API call, `DELETE /v1/ml-models/{model_id}` ([API reference](docs/API.md#model-downloads)).

---

## Self-Hosted Fonts

The web interface serves its two typefaces itself from `cognitive_ui/public/fonts/` (declared in `cognitive_ui/src/index.css`). Both are licensed under the SIL Open Font License, Version 1.1. The full licence text, with each font's copyright notice, is the `OFL.txt` next to the font files. The `.woff2` files, Latin and Latin Extended subsets, come from Fontsource (`@fontsource`, 5.3.0).

| Font | Copyright | Files in `cognitive_ui/public/fonts/` | Licence text |
|------|-----------|---------------------------------------|--------------|
| Inter | Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter) | `inter/inter-latin-{400,500,600,700}-normal.woff2`, `inter/inter-latin-ext-{400,500,600,700}-normal.woff2` | [inter/OFL.txt](cognitive_ui/public/fonts/inter/OFL.txt) |
| IBM Plex Mono | Copyright 2017 IBM Corp. | `ibm-plex-mono/ibm-plex-mono-latin-{400,500}-normal.woff2`, `ibm-plex-mono/ibm-plex-mono-latin-ext-{400,500}-normal.woff2` | [ibm-plex-mono/OFL.txt](cognitive_ui/public/fonts/ibm-plex-mono/OFL.txt) |

---

## Basemap and Imagery Credits

### Esri World Imagery

The **Satellite** basemap loads Esri World Imagery tiles from `server.arcgisonline.com` in the browser. The map credits them in its attribution control, which starts collapsed behind the (i) button, with this line (`cognitive_ui/src/components/map/basemaps.ts`):

```text
Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community
```

The **Streets** and **Terrain** basemaps load the MapLibre demo style (`https://demotiles.maplibre.org/style.json`), and the attribution control shows the attribution that style declares.

### NASA HLS

The bundled sample, `resources/kahovka_data/`, is two crops of NASA Harmonized Landsat Sentinel-2 (HLS) L30 scenes (see "Sample Data" in the [README](README.md#sample-data)). The interface credits it as "NASA HLS L30" in the landing page's image caption, and as "NASA HLS L30 (Landsat 8/9)" in the source line of result cards and in the map mode imagery panel. HLS S30 data is credited as "NASA HLS S30 (Sentinel-2)". The landing page's footer reads "Imagery: NASA HLS, Copernicus Sentinel-2".

### Copernicus Sentinel-2

Results computed from Sentinel-2 files or a Sentinel-2 region carry "Copernicus Sentinel-2" in their source line, and the map mode imagery panel describes the dataset as "Copernicus Sentinel-2 L2A · 10 m · revisit 5 days". Sentinel-2 is also credited in the landing page's footer. The same source line names "USGS Landsat 8/9" and "NASA MODIS" for those datasets (`cognitive_ui/src/components/chat/result/resultModel.ts`).

---

## Oblast Boundaries

The map draws the boundaries of Ukraine's oblasts from `cognitive_ui/public/ukraine_oblasts.json`. Sofiia Drozd, an author of the DT4LC paper, prepared the file from the subnational administrative boundaries of Ukraine published by OCHA on the Humanitarian Data Exchange (HDX).

- **Source:** OCHA, subnational administrative boundaries of Ukraine, on HDX (dataset page to be added)
- **License:** Creative Commons Attribution 3.0 IGO (CC BY-IGO 3.0), https://creativecommons.org/licenses/by/3.0/igo/. HDX's data licences: https://docs.humdata.org/about/data-licenses/
- **Attribution:** wherever the map draws the boundaries, its attribution control, behind the (i) button, includes this line (`cognitive_ui/src/components/map/OblastBoundaries.tsx`):

  ```text
  Oblast boundaries: OCHA / HDX, CC BY-IGO 3.0
  ```
