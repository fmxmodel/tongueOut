# Third-Party Notices — ARKit Avatar Viewer (Track B1, FLAME 2023 Open)

This web product (the rigged head `head_arkit.glb` + this three.js viewer, which loads and
drives it with MediaPipe FaceLandmarker) redistributes and/or builds upon the third-party
components listed below. Their required attributions, license texts, and notices are
reproduced here and surfaced in-app via the **"Credits / Licenses"** panel.

Verified against the shipped artifacts on 2026-07-04. Full verbatim license texts also ship
alongside this file under `licenses/` (`licenses/mediapipe-Apache-2.0-LICENSE.txt`,
`licenses/three.js-MIT-LICENSE.txt`).

> Scope note: this file covers the **shipped web product** — the GLB avatar and the browser
> viewer. Pod-side *build tools* used only to produce the GLB (PyTorch3D, OpenCV, Blender) are
> addressed in the last section as an explicit, recorded decision, not an omission.

---

## 1. FLAME 2023 (Open) head model — CC-BY-4.0

The 3D head geometry shipped in `head_arkit.glb` is derived from the **FLAME 2023 (Open)** head
model, which is licensed under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)**
license. CC-BY-4.0 requires appropriate credit, a link to the license, and an indication that
changes were made — all provided below and reproduced in the in-app Credits panel.

**Attribution / credit (as displayed in-product):**

> This product uses the FLAME 2023 (Open) head model by the Max Planck Institute for
> Intelligent Systems, licensed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/).
> The model was fit to an input image, re-textured, and rigged; these are modifications of the
> original. FLAME: Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero,
> 'Learning a model of facial shape and expression from 4D scans,' ACM Transactions on Graphics
> (SIGGRAPH Asia) 2017.

- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **License text / link:** https://creativecommons.org/licenses/by/4.0/
- **Changes made (CC-BY-4.0 requires indicating modification):** the FLAME 2023 Open head model
  was **fit to an input image, re-textured (per-subject image-baked albedo), and re-rigged with
  52 ARKit-named blendshape morph targets**. These are modifications of the original model.
- **Required citation (FLAME paper):** Tianye Li, Timo Bolkart, Michael J. Black, Hao Li,
  Javier Romero. "Learning a model of facial shape and expression from 4D scans." *ACM
  Transactions on Graphics (Proc. SIGGRAPH Asia)*, 2017.
- **Source / model license page:** https://flame.is.tue.mpg.de/modellicense.html

> Only the FLAME 2023 **Open** (CC-BY-4.0) shape/geometry release is used. The MPI FLAME
> **texture/appearance** space (CC-BY-NC-SA-4.0, non-commercial) and standard/2019/2020 FLAME
> (MPI academic, non-commercial) are **not** used in this product.

---

## 2. MediaPipe / `@mediapipe/tasks-vision` — Apache License 2.0

The face-tracking runtime is Google's **MediaPipe** (`@mediapipe/tasks-vision`, v0.10.14). This
product **redistributes MediaPipe binaries** — the FaceLandmarker WebAssembly runtime under
`mediapipe/wasm/*` (`vision_wasm_internal.{js,wasm}`, `vision_wasm_nosimd_internal.{js,wasm}`)
— so Apache-2.0 §4(a) requires that a copy of the license accompany the distribution. It is
bundled below and at `licenses/mediapipe-Apache-2.0-LICENSE.txt`.

- **Component:** MediaPipe Tasks Vision (FaceLandmarker) — `@mediapipe/tasks-vision@0.10.14`
- **Copyright:** Copyright 2019–2024 The MediaPipe Authors (Google LLC), `mediapipe@google.com`
- **License:** Apache License, Version 2.0
- **Upstream source of this license text:** `google-ai-edge/mediapipe` repository `LICENSE`
  (https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) — the npm tarball for
  `@mediapipe/tasks-vision` ships **no** LICENSE and **no** NOTICE file, so both were sourced
  directly from the upstream repository.

**NOTICE-file status (Apache-2.0 §4(d)):** As of the 2026-07-04 verification, the upstream
`google-ai-edge/mediapipe` repository ships an Apache-2.0 `LICENSE` file but **does not contain
a separate `NOTICE` text file** (confirmed: repository root listing has no `NOTICE`; a
repo-wide code search for a `NOTICE` file returned zero results). Because Apache-2.0 §4(d) only
obligates redistribution of a NOTICE file *if the upstream Work includes one*, there is **no
upstream NOTICE content to propagate** here. The §4(a) obligation (include a copy of the
License) **is** met by bundling the full Apache-2.0 text below. The attribution above (copyright
to The MediaPipe Authors / Google LLC) stands in for the customary NOTICE credit. **If a future
MediaPipe release adds a NOTICE file, its contents must be reproduced here before re-shipping.**

### Apache License 2.0 (full text, as shipped upstream by MediaPipe)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

===========================================================================
For files under tasks/cc/text/language_detector/custom_ops/utils/utf/
===========================================================================
/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
```

> The Lucent Technologies addendum above governs only text/language-detector UTF utilities
> under `tasks/cc/text/...`; it is reproduced because it is part of MediaPipe's upstream
> `LICENSE` file, even though those source files are not part of the FaceLandmarker vision
> WASM this product actually ships.

---

## 3. three.js — MIT License

The 3D viewer is built on **three.js** (`three@0.170.0`). The three.js code is bundled into the
app's JavaScript (`assets/index-*.js`), and its MIT license banner is preserved in that bundle
(`Copyright 2010-2024 Three.js Authors`). The full MIT notice is reproduced here and at
`licenses/three.js-MIT-LICENSE.txt`.

```
The MIT License

Copyright © 2010-2024 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 4. Pod-side build tools NOT redistributed (recorded decision, not an omission)

The following components are used only on the reconstruction/rigging **pod** to *produce* the
`head_arkit.glb` asset. They are **build tools**: their code and binaries are **not** included
in, linked into, or delivered by the shipped web product (the GLB + this viewer). Under each
license, the attribution/notice obligation is triggered by *redistribution* of that component's
code or binary — which the shipped product does **not** do. Their notices are therefore
**intentionally not required for this web product**, and this is recorded as an informed
decision so it reads as deliberate scoping, not a missed attribution.

| Pod build tool | License | Notice required in the shipped web product? |
|---|---|---|
| **PyTorch3D** (fit/bake rasterization & cameras) | BSD-3-Clause | **No** — not redistributed; the BSD-3 copyright + disclaimer would be required only if PyTorch3D code/binaries were shipped. |
| **OpenCV** (`opencv-python`; classical inpainting during texture bake) | Apache-2.0 | **No** — not redistributed; the Apache LICENSE + OpenCV NOTICE would be required only if OpenCV binaries were shipped (e.g. shipping the pod image). |
| **Blender** (`bpy`; headless glTF/GLB assembly & export) | GPL | **No** — Blender is a build tool that *processes* data (like a compiler); its GPL does not reach the GLB output, and the Blender binary is **not** part of the shipped product. Do **not** redistribute the Blender binary as part of the web product. |

**Trigger to revisit:** if the **pod image, pod pipeline code, or any of these binaries** is
ever itself redistributed (as opposed to just its GLB output), the corresponding notices
(PyTorch3D BSD-3, OpenCV Apache-2.0 LICENSE + NOTICE) must be added to that distribution.

---

## Summary

| Component | Role in shipped product | License | Obligation status |
|---|---|---|---|
| FLAME 2023 Open head model | 3D geometry in GLB | CC-BY-4.0 | Credit + license link + "changes made" + FLAME paper cite — **wired** (this file + in-app Credits panel) |
| MediaPipe `@mediapipe/tasks-vision` | Face tracking; WASM binaries redistributed | Apache-2.0 | License text bundled (§4(a)); no upstream NOTICE exists (§4(d) not triggered) — **wired** |
| three.js | Viewer/renderer; code bundled | MIT | Copyright + permission notice — **wired** (in-bundle banner + this file) |
| PyTorch3D / OpenCV / Blender | Pod build tools; NOT shipped | BSD-3 / Apache-2.0 / GPL | Not required for the web product (recorded decision) |
