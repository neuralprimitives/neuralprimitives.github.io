const NORMALIZATION_BY_SAMPLE_KEY = new Map();
const GROUPS = [];
let RUNNING = false;

const OBJ_LOADER = new THREE.OBJLoader();
const PLY_LOADER = new THREE.PLYLoader();

function makeViewer(container, ids) {
  const viewer = {
    container,
    scene: null,
    renderer: null,
    currentObj: null,
    loading: false,
    loaderEl: document.getElementById(ids.loader),
    progressEl: document.getElementById(ids.progress),
    placeholderEl: document.getElementById(ids.placeholder)
  };

  if (!container) return viewer;

  viewer.scene = new THREE.Scene();
  viewer.scene.background = new THREE.Color(0xffffff);
  viewer.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(5, 10, 5);
  viewer.scene.add(dirLight);

  viewer.renderer = new THREE.WebGLRenderer({ antialias: true });
  viewer.renderer.setPixelRatio(window.devicePixelRatio || 1);
  // Avoid initializing with 0x0; let sync/resize handle final CSS-driven size.
  const initW = Math.max(1, container.clientWidth);
  const initH = Math.max(1, container.clientHeight);
  viewer.renderer.setSize(initW, initH, false);
  const oldCanvas = container.querySelector("canvas");
  if (oldCanvas) oldCanvas.remove();
  viewer.renderer.domElement.style.display = "block";
  viewer.renderer.domElement.style.width = "100%";
  viewer.renderer.domElement.style.height = "100%";
  container.appendChild(viewer.renderer.domElement);

  return viewer;
}

function syncRendererSize(viewer) {
  if (!viewer || !viewer.container || !viewer.renderer) return false;
  const w = viewer.container.clientWidth;
  const h = viewer.container.clientHeight;
  if (w <= 0 || h <= 0) return false;

  const canvas = viewer.renderer.domElement;
  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.floor(w * dpr);
  const targetH = Math.floor(h * dpr);

  if (canvas.width !== targetW || canvas.height !== targetH) {
    viewer.renderer.setPixelRatio(dpr);
    viewer.renderer.setSize(w, h, false);
    return true;
  }
  return false;
}

function resetGroupView(group) {
  if (!group) return;

  // 恢复自动旋转
  group.autoRotate = true;

  // 恢复相机/controls 位置
  if (group.camera && group.defaultCameraPos) {
    group.camera.position.copy(group.defaultCameraPos);
  }
  if (group.controls && group.defaultTarget) {
    group.controls.target.copy(group.defaultTarget);
    group.controls.update();
  }

  // 清零每个对象的自转累计角度（否则会沿用旧角度）
  if (group.viewers) {
    for (const v of group.viewers) {
      if (!v || !v.currentObj) continue;
      v.currentObj.userData.autoRotateAngle = 0;

      // 关键：把当前姿态恢复成“归一化基姿态”
      // normalizeGroupWithNorm 会设置 baseQuaternion/scale/translation
      // 这里如果已经有 norm，就强制重新应用一次
      const sampleId = group.currentSample;
      if (sampleId) {
        const key = `${group.datasetDir}/${sampleId}`;
        const norm = NORMALIZATION_BY_SAMPLE_KEY.get(key);
        if (norm) normalizeGroupWithNorm(v.currentObj, norm);
      }
    }
  }
}



function clearObject(viewer) {
  if (!viewer || !viewer.scene || !viewer.currentObj) return;

  viewer.currentObj.traverse((child) => {
    if (child.isMesh || child.isPoints) {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of materials) {
          if (m && m.dispose) m.dispose();
        }
      }
    }
  });

  viewer.scene.remove(viewer.currentObj);
  viewer.currentObj = null;
}

function setLoading(viewer, isLoading) {
  if (viewer.loaderEl) viewer.loaderEl.style.display = isLoading ? "block" : "none";
  if (viewer.progressEl) viewer.progressEl.style.width = "0%";
  if (viewer.placeholderEl) viewer.placeholderEl.style.display = isLoading ? "none" : "block";
}


// function styleMesh(object3d, options = {}) {
//   const doubleSided = !!options.doubleSided;
//   object3d.traverse((child) => {
//     if (child.isMesh) {
//       if (child.geometry) child.geometry.computeVertexNormals();
//       child.material = new THREE.MeshStandardMaterial({
//         // color: 0xd8cab0,
//         // roughness: 0.7,
//         // metalness: 0.1,
//         // side: doubleSided ? THREE.DoubleSide : THREE.FrontSide
//         color: 0xA6A6A6 ,
//         // color: 0x8FA9C6,
//         roughness: 0.85,
//         metalness: 0.1,
//         side: doubleSided ? THREE.DoubleSide : THREE.FrontSide
//       });
//       child.frustumCulled = false;
//     }
//   });
// }

function styleMesh(object3d, options = {}) {
  const doubleSided = !!options.doubleSided;

  const grayHex = options.grayHex ?? 0xA6A6A6;
  const blueHex = options.blueHex ?? 0x9FB6D1;
  const roughness = options.roughness ?? 0.85;
  const metalness = options.metalness ?? 0.1;

  // “一个 face group”：蓝色三角形索引列表（triIndex）
  const blueTris = options.blueTris ?? null;

  function hexToRGB01(hex) {
    return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
  }

  object3d.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;

    const geom = child.geometry;
    console.log("[blueTris]", blueTris);
    // 没有 group：保持你原来的统一灰
    if (!Array.isArray(blueTris) || blueTris.length === 0) {
      // console.log("[test]", blueTris);
      geom.computeVertexNormals();
      child.material = new THREE.MeshStandardMaterial({
        color: grayHex,
        roughness,
        metalness,
        side: doubleSided ? THREE.DoubleSide : THREE.FrontSide
      });
      child.frustumCulled = false;
      return;
    } 
    // const effectiveBlueTris = Array.from({ length: 1000 }, (_, i) => i);

    // 有 group：按面上色（蓝/灰）
    const g = geom.index ? geom.toNonIndexed() : geom;
    const pos = g.attributes.position;
    const triCount = Math.floor(pos.count / 3);

    // triIndex 快速查表
    const isBlue = new Uint8Array(triCount);
    for (const t of blueTris) {
      if (t >= 0 && t < triCount) isBlue[t] = 1;
    }

    const colors = new Float32Array(pos.count * 3);
    const [gr, gg, gb] = hexToRGB01(grayHex);
    const [br, bg, bb] = hexToRGB01(blueHex);

    for (let tri = 0; tri < triCount; tri++) {
      const r = isBlue[tri] ? br : gr;
      const gC = isBlue[tri] ? bg : gg;
      const b = isBlue[tri] ? bb : gb;

      for (let k = 0; k < 3; k++) {
        const v = tri * 3 + k;
        colors[v * 3 + 0] = r;
        colors[v * 3 + 1] = gC;
        colors[v * 3 + 2] = b;
      }
    }

    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.computeVertexNormals();

    child.geometry = g;
    child.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness,
      metalness,
      side: doubleSided ? THREE.DoubleSide : THREE.FrontSide
    });
    child.frustumCulled = false;
  });
}



function centerAndScale(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = maxDim > 0 ? 0.9 / maxDim : 1.0;
  object3d.scale.setScalar(scale);

  const box2 = new THREE.Box3().setFromObject(object3d);
  const center = box2.getCenter(new THREE.Vector3());
  object3d.position.sub(center);
}

function resetTransform(object3d) {
  object3d.position.set(0, 0, 0);
  object3d.rotation.set(0, 0, 0);
  object3d.scale.set(1, 1, 1);
}

function getUpAlignmentQuaternion(datasetGroup, objectGroup) {
  const mode = (datasetGroup && datasetGroup.upAxis) || "y";
  const identity = new THREE.Quaternion();

  if (mode === "y") return identity;
  if (mode === "z") return new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));

  if (mode !== "auto" || !objectGroup) return identity;

  resetTransform(objectGroup);
  const box = new THREE.Box3().setFromObject(objectGroup);
  const size = box.getSize(new THREE.Vector3());

  // Heuristic: if the Z extent is significantly larger than Y, assume Z-up.
  if (Number.isFinite(size.z) && Number.isFinite(size.y) && size.z > size.y * 1.25) {
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
  }
  return identity;
}

function getMaxDim(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  const size = box.getSize(new THREE.Vector3());
  return Math.max(size.x, size.y, size.z);
}

function normalizeGroupWithNorm(objectGroup, norm) {
  if (!objectGroup || !norm) return;
  resetTransform(objectGroup);

  if (norm.quaternion) objectGroup.quaternion.copy(norm.quaternion);
  objectGroup.scale.setScalar(norm.scaleFactor);
  objectGroup.position.copy(norm.translation);

  objectGroup.userData.baseQuaternion = (norm.quaternion ? norm.quaternion.clone() : new THREE.Quaternion());
  if (!Number.isFinite(objectGroup.userData.autoRotateAngle)) objectGroup.userData.autoRotateAngle = 0;
}

function autoFrameGroupCamera(datasetGroup, referenceObject) {
  if (!datasetGroup || !datasetGroup.camera || !datasetGroup.controls || !referenceObject) return;

  const box = new THREE.Box3().setFromObject(referenceObject);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(maxDim) || maxDim <= 0) return;

  const targetHeightFactor = Number.isFinite(datasetGroup.autoFrameTargetHeightFactor)
    ? datasetGroup.autoFrameTargetHeightFactor
    : 0.35;

  // Aim a bit above the bottom so the object feels grounded.
  const target = new THREE.Vector3(center.x, box.min.y + size.y * targetHeightFactor, center.z);
  datasetGroup.controls.target.copy(target);

  // Fit bounding sphere into view.
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  const radius = Math.max(1e-6, sphere.radius);

  const fovRad = (datasetGroup.camera.fov * Math.PI) / 180;
  const distanceMultiplier = Number.isFinite(datasetGroup.autoFrameDistanceMultiplier)
    ? datasetGroup.autoFrameDistanceMultiplier
    : 1.15;

  const distance = (radius / Math.sin(fovRad / 2)) * distanceMultiplier;

  const fallbackDir = new THREE.Vector3(0.45, 0.25, 1.0);
  let dir = fallbackDir;
  const d = datasetGroup.autoFrameViewDir;
  if (d && typeof d === "object") {
    if (Array.isArray(d) && d.length >= 3) dir = new THREE.Vector3(d[0], d[1], d[2]);
    else if (Number.isFinite(d.x) && Number.isFinite(d.y) && Number.isFinite(d.z)) dir = new THREE.Vector3(d.x, d.y, d.z);
  }
  if (!Number.isFinite(dir.lengthSq()) || dir.lengthSq() < 1e-8) dir = fallbackDir;
  dir.normalize();
  datasetGroup.camera.position.copy(target).add(dir.multiplyScalar(distance));

  datasetGroup.camera.near = Math.max(0.001, distance / 200);
  datasetGroup.camera.far = Math.max(10, distance * 50);
  datasetGroup.camera.updateProjectionMatrix();
  datasetGroup.controls.update();
}

function maybeSetReferenceScale(sampleKey, sourceObjectGroup, datasetGroup) {
  if (!sampleKey || !sourceObjectGroup) return false;
  if (NORMALIZATION_BY_SAMPLE_KEY.has(sampleKey)) return false;

  const alignToGround = !!(datasetGroup && datasetGroup.alignToGround);

  const quaternion = getUpAlignmentQuaternion(datasetGroup, sourceObjectGroup);

  resetTransform(sourceObjectGroup);
  sourceObjectGroup.quaternion.copy(quaternion);
  const box = new THREE.Box3().setFromObject(sourceObjectGroup);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const scaleFactor = maxDim > 0 ? 0.9 / maxDim : 1.0;

  // Translation that centers the reference object's bbox center at origin.
  const refCenter = box.getCenter(new THREE.Vector3());
  const translation = refCenter.multiplyScalar(-scaleFactor);

  if (alignToGround) {
    // Compute ground offset for the reference object using the *same* translation.
    const tmp = sourceObjectGroup.clone(true);
    resetTransform(tmp);
    tmp.quaternion.copy(quaternion);
    tmp.scale.setScalar(scaleFactor);
    tmp.position.copy(translation);
    const box2 = new THREE.Box3().setFromObject(tmp);
    const minY = box2.min.y;
    if (Number.isFinite(minY)) translation.y += -minY;
  }

  NORMALIZATION_BY_SAMPLE_KEY.set(sampleKey, { scaleFactor, translation: translation.clone(), quaternion: quaternion.clone() });
  return true;
}

function applyNormalizationForSample(sampleKey, viewers) {
  if (!sampleKey) return;
  const norm = NORMALIZATION_BY_SAMPLE_KEY.get(sampleKey);
  if (!norm) return;

  for (const v of viewers) {
    if (!v || !v.currentObj) continue;
    normalizeGroupWithNorm(v.currentObj, norm);
  }
}


function buildCandidatePaths(datasetDir, sampleId, method, group, primitive_id = null) {
  let frame = frameStrFromSliderValue(group?.deviationValue ?? 0);
  if (primitive_id ) { frame = primitive_id; }
  const base = `./static/data/${datasetDir}/${sampleId}/${method}/${frame}`;
  return [`${base}.obj`, `${base}.ply`];

}


function loadColoredPointPlyInto(viewer, datasetGroup, sampleId, method) {
  if (!viewer || !viewer.scene || viewer.loading) return;
  viewer.loading = true;
  setLoading(viewer, true);
  clearObject(viewer);

  const path = `./static/data/${datasetGroup.datasetDir}/${sampleId}_${method}.ply`;

  PLY_LOADER.load(
    path,
    (geometry) => {
      geometry.computeVertexNormals?.();

      const hasColors = !!(geometry && geometry.getAttribute && geometry.getAttribute("color"));
      const material = new THREE.PointsMaterial({
        size: 0.01,
        color: hasColors ? 0xffffff : 0xA6A6A6,
        vertexColors: hasColors
      });

      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;

      const objectGroup = new THREE.Group();
      objectGroup.add(points);
      viewer.scene.add(objectGroup);
      viewer.currentObj = objectGroup;

      const sampleKey = `${datasetGroup.datasetDir}/${sampleId}`;
      const norm = NORMALIZATION_BY_SAMPLE_KEY.get(sampleKey);
      if (norm) {
        normalizeGroupWithNorm(objectGroup, norm);
      } else {
        resetTransform(objectGroup);
        const maxDim = getMaxDim(objectGroup);
        const tmpScale = maxDim > 0 ? 0.9 / maxDim : 1.0;
        const box = new THREE.Box3().setFromObject(objectGroup);
        const center = box.getCenter(new THREE.Vector3());
        normalizeGroupWithNorm(objectGroup, { scaleFactor: tmpScale, translation: center.multiplyScalar(-tmpScale) });
      }

      viewer.loading = false;
      setLoading(viewer, false);
    },
    undefined,
    (err) => {
      console.warn("Failed to load", path, err);
      if (viewer.loaderEl) viewer.loaderEl.innerHTML = "<div style='color:red;'>Load failed</div>";
      viewer.loading = false;
      setLoading(viewer, false);
    }
  );
}

// async function loadModelInto(viewer, group, sampleId, method, primitive_id = null) {
//   if (!viewer || !viewer.scene || viewer.loading) return;
//   const useToken = (viewer.role === "primitive");
// const expectedToken = useToken ? (viewer._activeLoadToken ?? 0) : 0;
//   viewer.loading = true;
//   setLoading(viewer, true);
//   clearObject(viewer);

//   const candidates = buildCandidatePaths(group.datasetDir, sampleId, method, group, primitive_id);
//   let blueTris = null;

//   if (useToken) {
//     blueTris = await getBlueFaceListForPrimitive(
//       group.datasetDir,
//       sampleId,
//       method,
//       group,
//       primitive_id
//     );
//   }
  


//   const sampleKey = `${group.datasetDir}/${sampleId}`;

//   const tryLoad = (idx) => {
//     if (idx >= candidates.length) {
//       console.error(`No loadable asset found for ${sampleId}_${method}`);
//       if (viewer.loaderEl) viewer.loaderEl.innerHTML = "<div style='color:red;'>Load failed</div>";
//       viewer.loading = false;
//       setLoading(viewer, false);
//       return;
//     }

//     const path = candidates[idx];
//     const isPLY = path.toLowerCase().endsWith(".ply");

//     const onError = (err) => {
//       console.warn("Failed to load", path, err);
//       tryLoad(idx + 1);
//     };

//     if (isPLY) {
//       if (useToken && (viewer._activeLoadToken ?? 0) !== expectedToken) {
//   viewer.loading = false;
//   setLoading(viewer, false);
//   return;
// }
//       PLY_LOADER.load(
//         path,
//         (geometry) => {
//           geometry.computeVertexNormals?.();
//           const material = new THREE.PointsMaterial({
//             size: 0.01,
//             color: 0xA6A6A6
//           });
//           const points = new THREE.Points(geometry, material);

//           const objectGroup = new THREE.Group();
//           objectGroup.add(points);
//           // Normalize later using shared per-sample scale.
//           viewer.scene.add(objectGroup);
//           viewer.currentObj = objectGroup;

//           // If we already have a reference scale, apply it; otherwise do a temporary self-scale.
//           const norm = NORMALIZATION_BY_SAMPLE_KEY.get(sampleKey);
//           if (norm) {
//             normalizeGroupWithNorm(objectGroup, norm);
//           } else {
//             const quaternion = getUpAlignmentQuaternion(group, objectGroup);
//             resetTransform(objectGroup);
//             objectGroup.quaternion.copy(quaternion);
//             const maxDim = getMaxDim(objectGroup);
//             const tmpScale = maxDim > 0 ? 0.9 / maxDim : 1.0;
//             const box = new THREE.Box3().setFromObject(objectGroup);
//             const center = box.getCenter(new THREE.Vector3());
//             normalizeGroupWithNorm(objectGroup, {
//               scaleFactor: tmpScale,
//               translation: center.multiplyScalar(-tmpScale),
//               quaternion
//             });
//           }

//           // Prefer GT as reference; fallback to vecunico.
//           if ((viewer.role === "gt" || viewer.role === "vecunico") && maybeSetReferenceScale(sampleKey, objectGroup, viewer.group)) {
//             applyNormalizationForSample(sampleKey, viewer.group ? viewer.group.viewers : []);
//             if (viewer.group && viewer.group.autoFrame) {autoFrameGroupCamera(viewer.group, objectGroup);
//               freezeDefaultViewFromCurrent(viewer.group);
//             }
//           }

//           viewer.loading = false;
//           setLoading(viewer, false);
//         },
//         undefined,
//         onError
//       );
//     } else {
//       if (useToken && (viewer._activeLoadToken ?? 0) !== expectedToken) {
//   viewer.loading = false;
//   setLoading(viewer, false);
//   return;
// }
//       OBJ_LOADER.load(
//         path,
//         (obj) => {
//           styleMesh(obj, { doubleSided: !!group.doubleSided, blueTris: blueTris });
//           const objectGroup = new THREE.Group();
//           objectGroup.add(obj);
//           viewer.scene.add(objectGroup);
//           viewer.currentObj = objectGroup;

//           const norm = NORMALIZATION_BY_SAMPLE_KEY.get(sampleKey);
//           if (norm) {
//             normalizeGroupWithNorm(objectGroup, norm);
//           } else {
//             const quaternion = getUpAlignmentQuaternion(group, objectGroup);
//             resetTransform(objectGroup);
//             objectGroup.quaternion.copy(quaternion);
//             const maxDim = getMaxDim(objectGroup);
//             const tmpScale = maxDim > 0 ? 0.9 / maxDim : 1.0;
//             const box = new THREE.Box3().setFromObject(objectGroup);
//             const center = box.getCenter(new THREE.Vector3());
//             normalizeGroupWithNorm(objectGroup, {
//               scaleFactor: tmpScale,
//               translation: center.multiplyScalar(-tmpScale),
//               quaternion
//             });
//           }

//           if ((viewer.role === "gt" || viewer.role === "vecunico") && maybeSetReferenceScale(sampleKey, objectGroup, viewer.group)) {
//             applyNormalizationForSample(sampleKey, viewer.group ? viewer.group.viewers : []);
//             if (viewer.group && viewer.group.autoFrame) {
//               autoFrameGroupCamera(viewer.group, objectGroup);
//               freezeDefaultViewFromCurrent(viewer.group);
//             }
//           }

//           viewer.loading = false;
//           setLoading(viewer, false);
//         },
//         undefined,
//         onError
//       );
//     }
//   };

//   tryLoad(0);
// }
function loadModelInto(viewer, group, sampleId, method, primitive_id = null) {
  return new Promise(async (resolve) => {
    if (!viewer || !viewer.scene || viewer.loading) return resolve({ ok: false, reason: "skip" });

    const useToken = (viewer.role === "primitive");
    const expectedToken = useToken ? (viewer._activeLoadToken ?? 0) : 0;

    const tokenCanceled = () => useToken && (viewer._activeLoadToken ?? 0) !== expectedToken;

    viewer.loading = true;
    setLoading(viewer, true);
    clearObject(viewer);

    const finish = (ok, reason = "") => {
      viewer.loading = false;
      setLoading(viewer, false);
      resolve({ ok, reason });
    };

    const candidates = buildCandidatePaths(group.datasetDir, sampleId, method, group, primitive_id);

    // primitive highlight faces
    let blueTris = null;
    if (useToken) {
      blueTris = await getBlueFaceListForPrimitive(
        group.datasetDir,
        sampleId,
        method,
        group,
        primitive_id
      );
      if (tokenCanceled()) return finish(false, "canceled");
    }

    const sampleKey = `${group.datasetDir}/${sampleId}`;

    const applyLocalNormalize = (objectGroup) => {
      const norm = NORMALIZATION_BY_SAMPLE_KEY.get(sampleKey);
      if (norm) {
        normalizeGroupWithNorm(objectGroup, norm);
      } else {
        const quaternion = getUpAlignmentQuaternion(group, objectGroup);
        resetTransform(objectGroup);
        objectGroup.quaternion.copy(quaternion);
        const maxDim = getMaxDim(objectGroup);
        const tmpScale = maxDim > 0 ? 0.9 / maxDim : 1.0;
        const box = new THREE.Box3().setFromObject(objectGroup);
        const center = box.getCenter(new THREE.Vector3());
        normalizeGroupWithNorm(objectGroup, {
          scaleFactor: tmpScale,
          translation: center.multiplyScalar(-tmpScale),
          quaternion
        });
      }
    };

    const tryLoad = (idx) => {
      if (idx >= candidates.length) {
        console.error(`No loadable asset found for ${sampleId}_${method}`);
        if (viewer.loaderEl) viewer.loaderEl.innerHTML = "<div style='color:red;'>Load failed</div>";
        return finish(false, "no_asset");
      }

      const path = candidates[idx];
      const isPLY = path.toLowerCase().endsWith(".ply");

      const onError = (err) => {
        console.warn("Failed to load", path, err);
        tryLoad(idx + 1);
      };

      if (tokenCanceled()) return finish(false, "canceled");

      if (isPLY) {
        PLY_LOADER.load(
          path,
          (geometry) => {
            if (tokenCanceled()) return finish(false, "canceled");

            geometry.computeVertexNormals?.();
            const material = new THREE.PointsMaterial({ size: 0.01, color: 0xA6A6A6 });
            const points = new THREE.Points(geometry, material);

            const objectGroup = new THREE.Group();
            objectGroup.add(points);
            viewer.scene.add(objectGroup);
            viewer.currentObj = objectGroup;

            applyLocalNormalize(objectGroup);

            // ❌ 不要在这里做 applyNormalizationForSample/autoFrame/freeze
            return finish(true, "ok");
          },
          undefined,
          onError
        );
      } else {
        OBJ_LOADER.load(
          path,
          (obj) => {
            if (tokenCanceled()) return finish(false, "canceled");

            styleMesh(obj, { doubleSided: !!group.doubleSided, blueTris });

            const objectGroup = new THREE.Group();
            objectGroup.add(obj);
            viewer.scene.add(objectGroup);
            viewer.currentObj = objectGroup;

            applyLocalNormalize(objectGroup);

            // ❌ 不要在这里做 applyNormalizationForSample/autoFrame/freeze
            return finish(true, "ok");
          },
          undefined,
          onError
        );
      }
    };

    tryLoad(0);
  });
}

function setGroupInteractionEnabled(group, enabled) {
  const viewers = [
    group.gtViewer,
    group.inputViewer,
    group.vecunicoViewer,
    group.primitiveViewer,
    group.methodViewer,
    group.vecunicoPointsViewer,
    group.methodPointsViewer
  ].filter(Boolean);

  viewers.forEach(v => {
    if (v?.controls) {
      v.controls.enabled = enabled;
      if (!enabled) v.controls.autoRotate = false; // loading期间别乱转
    }
  });
}

function freezeDefaultViewFromCurrent(group) {
  if (!group || !group.camera || !group.controls) return;

  // 只冻结一次，避免后续异步 load 又覆盖
  if (group.__defaultFrozen) return;

  group.defaultCameraPos = group.camera.position.clone();
  group.defaultTarget = group.controls.target.clone();
  group.__defaultFrozen = true;

  // 额外稳：OrbitControls 自带 state，可用于 reset()
  if (typeof group.controls.saveState === "function") {
    group.controls.saveState();
  }
}

function initCameraAndControls(group, domElementForControls) {
  const fov = Number.isFinite(group.cameraFov) ? group.cameraFov : 45;
  group.camera = new THREE.PerspectiveCamera(fov, 1.0, 0.1, 1000);
  // Slightly farther away so objects appear smaller by default.
  group.camera.position.set(0.5, 0.5, 1.8);
  group.camera.lookAt(0, 0, 0);

  group.controls = new THREE.OrbitControls(group.camera, domElementForControls);
  group.controls.enableDamping = true;
  group.controls.target.set(0, 0, 0);
  group.controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN
  };
  group.controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN
  };
  let resumeTimer = null;

  group.controls.addEventListener("start", () => {
    group.autoRotate = false;
    if (resumeTimer) clearTimeout(resumeTimer);
  });

  group.controls.addEventListener("end", () => {
    if (resumeTimer) clearTimeout(resumeTimer);
    resumeTimer = setTimeout(() => {
      group.autoRotate = true;
      // 可选：清零角度，让它从当前姿态继续转
      for (const v of group.viewers) if (v.currentObj) v.currentObj.userData.autoRotateAngle = 0;
    }, 800); // 0.8s 后恢复
  });
  // group.defaultCameraPos = group.camera.position.clone();
  // group.defaultTarget = group.controls.target.clone();

  group.controls.update();
  group.controls.addEventListener("start", () => {
    group.autoRotate = false;
  });
}

function bindControlsToDom(group, domElementForControls) {
  if (!group || !group.camera || !domElementForControls) return;
  if (group.controls && group.controls.domElement === domElementForControls) return;

  const prevTarget = group.controls ? group.controls.target.clone() : new THREE.Vector3(0, 0, 0);
  const prevEnableDamping = group.controls ? group.controls.enableDamping : true;
  const prevDampingFactor = group.controls ? group.controls.dampingFactor : 0.05;
  const prevRotateSpeed = group.controls ? group.controls.rotateSpeed : 1.0;
  const prevZoomSpeed = group.controls ? group.controls.zoomSpeed : 1.0;
  const prevPanSpeed = group.controls ? group.controls.panSpeed : 1.0;

  if (group.controls) group.controls.dispose();

  group.controls = new THREE.OrbitControls(group.camera, domElementForControls);
  group.controls.enableDamping = prevEnableDamping;
  group.controls.dampingFactor = prevDampingFactor;
  group.controls.rotateSpeed = prevRotateSpeed;
  group.controls.zoomSpeed = prevZoomSpeed;
  group.controls.panSpeed = prevPanSpeed;
  group.controls.target.copy(prevTarget);
  group.controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN
  };
  group.controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN
  };
  group.controls.update();
  group.controls.addEventListener("start", () => {
    group.autoRotate = false;
  });
}

function handleResize(viewers) {
  for (const v of viewers) {
    if (!v || !v.container || !v.renderer) continue;
    const w = v.container.clientWidth;
    const h = v.container.clientHeight;
    if (w <= 0 || h <= 0) continue;
    v.renderer.setSize(w, h, false);
  }
}

function animateGroups() {
  if (!RUNNING) return;
  requestAnimationFrame(() => animateGroups());

  for (const group of GROUPS) {
    if (!group || !group.viewers || !group.camera) continue;

    if (group.controls) group.controls.update();

    for (const v of group.viewers) syncRendererSize(v);

    if (group.autoRotate) {
  const axisLocal = group.autoRotateAxisLocal === "z"
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(0, 1, 0);

  // ✅ 共享角度：整个 group 只有一个旋转相位
  if (!Number.isFinite(group._autoRotateAngle)) group._autoRotateAngle = 0;
  group._autoRotateAngle += 0.01;

  for (const v of group.viewers) {
    if (!v || !v.currentObj) continue;
    const obj = v.currentObj;

    const baseQ = obj.userData.baseQuaternion
      ? obj.userData.baseQuaternion.clone()
      : new THREE.Quaternion();

    const axisWorld = axisLocal.clone().applyQuaternion(baseQ).normalize();

    const rotQ = new THREE.Quaternion().setFromAxisAngle(axisWorld, group._autoRotateAngle);
    obj.quaternion.copy(rotQ).multiply(baseQ);
  }
}

    for (const v of group.viewers) {
      if (!v || !v.renderer || !v.scene) continue;
      const w = v.container.clientWidth;
      const h = v.container.clientHeight;
      if (w <= 0 || h <= 0) continue;
      group.camera.aspect = w / h;
      group.camera.updateProjectionMatrix();
      v.renderer.render(v.scene, group.camera);
    }
  }
}

function setActiveButton(buttons, activeBtn) {
  for (const b of buttons) {
    b.classList.remove("is-primary");
    b.classList.remove("is-light");
  }
  if (activeBtn) {
    activeBtn.classList.add("is-primary");
    activeBtn.classList.add("is-light");
  }
}


function bindSelectorsForGroup(group, sampleContainerId, methodContainerId) {
  const sampleContainer = document.getElementById(sampleContainerId);
  const methodContainer = document.getElementById(methodContainerId);
  if (!methodContainer) return;

  const sampleButtons = Array.from(sampleContainer.querySelectorAll("[data-sample]"));
  const methodButtons = Array.from(methodContainer.querySelectorAll("[data-method]"));

  // const applyLoad = () => {
  //   if (!group.currentSample) return;
  //   loadModelInto(group.gtViewer, group, group.currentSample, "gt");
  //   loadModelInto(group.inputViewer, group, group.currentSample, "input");
  //   loadModelInto(group.vecunicoViewer, group, group.currentSample, "vecunico");
  //   if (group.primitiveViewer) loadModelInto(group.primitiveViewer, group, group.currentSample, "primitive");
  //   if (group.methodViewer) loadModelInto(group.methodViewer, group, group.currentSample, group.currentMethod);

  //   if (group.vecunicoPointsViewer) loadColoredPointPlyInto(group.vecunicoPointsViewer, group, group.currentSample, "vecunico");
  //   if (group.methodPointsViewer) loadColoredPointPlyInto(group.methodPointsViewer, group, group.currentSample, group.currentMethod);
  // };

  for (const btn of sampleButtons) {
    btn.addEventListener("click", () => {
      group.currentSample = btn.getAttribute("data-sample");
      setActiveButton(sampleButtons, btn);
      
      group.applyLoad();
    });
  }

  for (const btn of methodButtons) {
    btn.addEventListener("click", () => {
      group.currentMethod = btn.getAttribute("data-method") || group.currentMethod;
      setActiveButton(methodButtons, btn);
      const methodTitle =
      btn.getAttribute("data-method-title") || group.currentMethod;

      setActiveButton(methodButtons, btn);

      
      if (group.methodTitleEl) {
        group.methodTitleEl.textContent = methodTitle;
      }
      resetGroupView(group);
      group.applyLoad();
    });
  }
  
    // if (!group.currentSample) return;

    // const sampleId = group.currentSample;
    // const method = group.currentMethod;

    // // 只更新 method（以及 method points）
    // loadModelInto(group.methodViewer, group, group.currentSample, group.currentMethod);
    // if (group.methodPointsViewer) {
    //   loadColoredPointPlyInto(group.methodPointsViewer, group, sampleId, method);
    // }

    // // （可选）重置自转角即可，不必动 camera/target
    // // if (group.methodViewer?.currentObj) group.methodViewer.currentObj.userData.autoRotateAngle = 0;
    // resetGroupView(group);
//   });
// }

  const defaultSampleBtn = sampleButtons.find((b) => b.getAttribute("data-default") === "true") || sampleButtons[0];
  const defaultMethodBtn = methodButtons.find((b) => b.getAttribute("data-default") === "true") || methodButtons[0];

  if (defaultMethodBtn) defaultMethodBtn.click();
  if (defaultSampleBtn) defaultSampleBtn.click();
}


function bindDeviationSliderForGroup(group, sliderId) {
  const slider = document.getElementById(sliderId);
  if (!slider || !group) return;

  const onChange = () => {
    group.deviationValue = parseInt(slider.value, 10) || 0;
    resetGroupView(group);
    group.applyLoad();
  };

  slider.addEventListener("input", onChange);   // 拖动实时更新
  slider.addEventListener("change", onChange);  // 兼容
  onChange(); // 初始化：用 slider 初始值触发一次（通常是 0 -> "000"）
}



function initGroup(config) {
  const group = {
    datasetDir: config.datasetDir,
    defaultCameraPos: new THREE.Vector3(0.5, 0.5, 1.8),
    defaultTarget: new THREE.Vector3(0, 0, 0),
    alignToGround: !!config.alignToGround,
    upAxis: config.upAxis || "y",
    doubleSided: !!config.doubleSided,
    autoRotateAxisLocal: config.autoRotateAxisLocal || "y",
    autoFrame: config.autoFrame !== false,
    autoFrameViewDir: config.autoFrameViewDir,
    autoFrameTargetHeightFactor: Number.isFinite(config.autoFrameTargetHeightFactor) ? config.autoFrameTargetHeightFactor : undefined,
    autoFrameDistanceMultiplier: Number.isFinite(config.autoFrameDistanceMultiplier) ? config.autoFrameDistanceMultiplier : undefined,
    cameraFov: Number.isFinite(config.cameraFov) ? config.cameraFov : undefined,
    autoRotate: true,
    currentSample: null,
    currentMethod: config.defaultMethod,
    camera: null,
    controls: null,
    viewers: [],
    gtViewer: null,
    inputViewer: null,
    vecunicoViewer: null,
    methodViewer: null,
    methodTitleEl: null,
    primitiveViewer: null,
    vecunicoPointsViewer: null,
    methodPointsViewer: null
  };

  const gtViewer = makeViewer(document.getElementById(config.viewerIds.gt), config.viewerUI.gt);
  if (!gtViewer.container) return null;
  gtViewer.role = "gt";

  const inputViewer = makeViewer(document.getElementById(config.viewerIds.input), config.viewerUI.input);
  inputViewer.role = "input";

  const vecunicoViewer = makeViewer(document.getElementById(config.viewerIds.vecunico), config.viewerUI.vecunico);
  vecunicoViewer.role = "vecunico";

  group.gtViewer = gtViewer;
  group.inputViewer = inputViewer;
  group.vecunicoViewer = vecunicoViewer;
  

  const viewers = [gtViewer, inputViewer, vecunicoViewer];
  let primitiveCarouselIds = [];

  if (config.viewerIds.method) {
    const methodViewer = makeViewer(
      document.getElementById(config.viewerIds.method),
      config.viewerUI.method
    );
    methodViewer.role = "method";
    group.methodViewer = methodViewer;

    group.methodTitleEl =
      document.getElementById(config.viewerIds.method)
        .querySelector("[data-method-title]");

    viewers.push(methodViewer);
  } else if (config.viewerIds.primitive) {


    const primitiveViewer = makeViewer(
      document.getElementById(config.viewerIds.primitive),
      config.viewerUI.primitive
    );
    primitiveViewer.role = "primitive";
    primitiveViewer.metaEl = document.createElement("div");
    primitiveViewer.metaEl.className = "primitive-meta";
    primitiveViewer.container.style.position = "relative";
    primitiveViewer.container.appendChild(primitiveViewer.metaEl);
    primitiveViewer.metaEl.style.display = "none"; // 默认隐藏

    group.primitiveViewer = primitiveViewer;

    viewers.push(primitiveViewer) ;
    
  }

  

  if (config.viewerIds.vecunicoPoints && config.viewerIds.methodPoints) {
    const vecunicoPointsViewer = makeViewer(document.getElementById(config.viewerIds.vecunicoPoints), config.viewerUI.vecunicoPoints);
    const methodPointsViewer = makeViewer(document.getElementById(config.viewerIds.methodPoints), config.viewerUI.methodPoints);
    if (vecunicoPointsViewer.container && methodPointsViewer.container) {
      vecunicoPointsViewer.role = "vecunico_points";
      methodPointsViewer.role = "method_points";
      group.vecunicoPointsViewer = vecunicoPointsViewer;
      group.methodPointsViewer = methodPointsViewer;
      viewers.push(vecunicoPointsViewer, methodPointsViewer);
    }
  }

  group.viewers = viewers;
  // attach group reference for normalization propagation
  for (const v of viewers) v.group = group;
  // ==============================
// Primitive carousel (optional)
// ==============================
group.startPrimitiveCarousel = function(primitiveids, {
  intervalMs = 10000,
  loop = true,
  shuffle = false
} = {}) {
  group.stopPrimitiveCarousel();

  if (!group.primitiveViewer) {
    console.warn("[carousel] primitiveViewer not available");
    return;
  }
  if (!Array.isArray(primitiveids) || primitiveids.length === 0) {
    console.warn("[carousel] empty primitiveids");
    return;
  }

  const ids = primitiveids.slice();
  if (shuffle) ids.sort(() => Math.random() - 0.5);

  group._primitiveCarousel = {
    ids,
    idx: 0,
    timer: null,
    token: 0,
    running: true,
    intervalMs,
    loop
  };

  const tick = () => {
    const c = group._primitiveCarousel;
    if (!c || !c.running) return;

    const v = group.primitiveViewer;
    // if (!v || v.loading) return;
    if (!v) return; if (v.loading) { setTimeout(tick, 50); return; }

    if (c.idx >= c.ids.length) {
      if (!c.loop) { group.stopPrimitiveCarousel(); return; }
      c.idx = 0;
    }

    const primitive_id = c.ids[c.idx++];
    const sampleId = group.currentSample;

    // ✅ token only for primitive
    c.token += 1;
    v._activeLoadToken = c.token;

    loadModelInto(v, group, sampleId, "primitive", primitive_id);
    updatePrimitiveMetaPanel(v, sampleId, primitive_id);
  };

  tick();
  group._primitiveCarousel.timer = setInterval(tick, intervalMs);
};

group.stopPrimitiveCarousel = function() {
  const c = group._primitiveCarousel;
  if (c?.timer) clearInterval(c.timer);
  if (c) c.running = false;
  group._primitiveCarousel = null;
};



  // ✅ unified loader (IMPORTANT: must be AFTER v.group is set)
  // group.applyLoad = function applyLoadAll() {
  //   if (!group.currentSample) return;

  //   const sampleId = group.currentSample;
  //   const method = group.currentMethod || config.defaultMethod || "paco";

  //   loadModelInto(group.gtViewer, group, sampleId, "gt");
  //   loadModelInto(group.inputViewer, group, sampleId, "input");
  //   loadModelInto(group.vecunicoViewer, group, sampleId, "vecunico");
    
  //   if (group.primitiveViewer) {
  //    loadModelInto(group.primitiveViewer, group, sampleId, "primitive");
  //   } 
  //   if (group.methodViewer)
  //       {
  //       loadModelInto(group.methodViewer, group, sampleId, method);
  //        // disable primitive carousel if method viewer is used
  //     }

  //   if (group.vecunicoPointsViewer) {
  //     loadColoredPointPlyInto(group.vecunicoPointsViewer, group, sampleId, "vecunico");
  //   }
  //   if (group.methodPointsViewer) {
  //     loadColoredPointPlyInto(group.methodPointsViewer, group, sampleId, method);
  //   }
  // };
  group.applyLoad = async function applyLoadAll() {
  if (!group.currentSample) return;

  // token to avoid stale load finishing later and messing camera/normalization
  group._applyLoadToken = (group._applyLoadToken ?? 0) + 1;
  const myToken = group._applyLoadToken;

  const sampleId = group.currentSample;
  const method = group.currentMethod || config.defaultMethod || "paco";
  const sampleKey = `${group.datasetDir}/${sampleId}`;

  setGroupInteractionEnabled(group, false);

  // fire all loads and WAIT
  const tasks = [
    loadModelInto(group.gtViewer, group, sampleId, "gt"),
    loadModelInto(group.inputViewer, group, sampleId, "input"),
    loadModelInto(group.vecunicoViewer, group, sampleId, "vecunico"),
  ];

  if (group.primitiveViewer) {
    tasks.push(loadModelInto(group.primitiveViewer, group, sampleId, "primitive"));
  }
  if (group.methodViewer) {
    tasks.push(loadModelInto(group.methodViewer, group, sampleId, method));
  }

  if (group.vecunicoPointsViewer) {
    tasks.push(loadColoredPointPlyInto(group.vecunicoPointsViewer, group, sampleId, "vecunico"));
  }
  if (group.methodPointsViewer) {
    tasks.push(loadColoredPointPlyInto(group.methodPointsViewer, group, sampleId, method));
  }

  await Promise.allSettled(tasks);

  // stale guard
  if (myToken !== group._applyLoadToken) return;

  // ✅ one-time reference & normalization (do it ONCE)
  // prefer GT as reference; fallback to vecunico
  const refObj =
    group.gtViewer?.currentObj ||
    group.vecunicoViewer?.currentObj ||
    group.methodViewer?.currentObj;

  if (refObj && maybeSetReferenceScale(sampleKey, refObj, group)) {
    applyNormalizationForSample(sampleKey, group.viewers ?? []);
  }

  // ✅ one-time auto-frame & freeze (do it ONCE)
  if (group.autoFrame && refObj) {
    autoFrameGroupCamera(group, refObj);
    freezeDefaultViewFromCurrent(group);
  }

  setGroupInteractionEnabled(group, true);
};

  // camera/controls
  initCameraAndControls(group, vecunicoViewer.renderer.domElement);

  // bind orbit controls to whichever canvas is hovered/clicked
  for (const v of viewers) {
    const canvas = v.renderer?.domElement;
    if (!canvas) continue;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", () => bindControlsToDom(group, canvas), true);
    canvas.addEventListener("pointerenter", () => bindControlsToDom(group, canvas));
    canvas.style.touchAction = "none";
  }

  handleResize(viewers);
  requestAnimationFrame(() => handleResize(viewers));

  // optional: your old sample/method selectors (if you still use them)
  if (config.sampleContainerId && config.methodContainerId) {
    bindSelectorsForGroup(group, config.sampleContainerId, config.methodContainerId);
  } else if (config.methodContainerId) {
    // 如果你现在 sample 由 thumbnail 控制，只绑 method
    bindSelectorsForGroup(group, null, config.methodContainerId);
  } 


  return group;
}

function selectThumb(el, group, thumbsContainerId, placeholderId) {
  if (!el || !group) return;

  const container = document.getElementById(thumbsContainerId);
  if (!container) return;

  const thumbs = Array.from(container.querySelectorAll("[data-sample]"));

  // UI state
  thumbs.forEach(t => t.classList.remove("thumbnail-selected"));
  el.classList.add("thumbnail-selected");

  // optional video behavior
  if (el.tagName && el.tagName.toLowerCase() === "video") el.play().catch(()=>{});
  container.querySelectorAll("video").forEach(vid => {
    if (vid !== el) { vid.pause(); vid.currentTime = 0; }
  });

  // Update group state
  group.currentSample = el.getAttribute("data-sample");
  const methodFromThumb = el.getAttribute("data-method");
  if (methodFromThumb) group.currentMethod = methodFromThumb; // optional
  resetGroupView(group);

  // Load all 4 viewers (so building grid stays consistent)
  // loadModelInto(group.gtViewer, group, group.currentSample, "gt");
  // loadModelInto(group.inputViewer, group, group.currentSample, "input");
  // loadModelInto(group.vecunicoViewer, group, group.currentSample, "vecunico");
  // if (group.primitiveViewer) {

  //   loadModelInto(group.primitiveViewer, group, group.currentSample, "primitive");
  // }
  // if (group.methodViewer){
  //   loadModelInto(group.methodViewer, group, group.currentSample, group.currentMethod);
  // }
  group.applyLoad();
  primitiveCarouselIds = getPrimitiveCarouselIds(group, group.currentSample);

  if (group.primitiveViewer && Array.isArray(primitiveCarouselIds) && primitiveCarouselIds.length > 0) {
  // 等一帧：确保 renderer/canvas/resize 都 ready
  requestAnimationFrame(() => {
    group.startPrimitiveCarousel(primitiveCarouselIds, {
      intervalMs: 10000,
      loop: true,
      shuffle: false
    });
  });
}
  // hide placeholder (optional)
  if (placeholderId) {
    const ph = document.getElementById(placeholderId);
    if (ph) ph.style.display = "none";
  }
}


function bindThumbnailsForGroup(group, thumbsContainerId, placeholderId) {
  const container = document.getElementById(thumbsContainerId);
  if (!container || !group) return;

  const thumbs = Array.from(container.querySelectorAll("[data-sample]"));
  if (thumbs.length === 0) return;

  // click binding
  for (const el of thumbs) {
    el.addEventListener("click", () => selectThumb(el, group, thumbsContainerId, placeholderId));
  }

  // default: data-default="true" first, else first thumb
  const defaultEl = thumbs.find(t => t.getAttribute("data-default") === "true") || thumbs[0];
  selectThumb(defaultEl, group, thumbsContainerId, placeholderId);
}


function frameStrFromSliderValue(v) {
  const n = Math.max(0, Math.min(999, parseInt(v, 10) || 0));
  return String(n).padStart(3, "0"); // 0 -> "000", 20 -> "020"
}



window.addEventListener("DOMContentLoaded", () => {

  const buildingGroup = initGroup({
    datasetDir: "building",
    defaultMethod: "paco",
    useDeviation: true,
    alignToGround: true,
    upAxis: "z",
    doubleSided: true,
    autoRotateAxisLocal: "z",
    cameraFov: 35,
    autoFrame: true,
    autoFrameViewDir: [0.0, 0.5, 0.5],
    autoFrameTargetHeightFactor: 0.35,
    autoFrameDistanceMultiplier: 1.1,

    sampleContainerId: "thumbnail-building",
    methodContainerId: "comparison-methods-building",

    viewerIds: {
      input: "input-viewer-building",
      gt: "gt-viewer-building",
      vecunico: "vecunico-viewer-building",
      method: "method-viewer-building"
    },
    viewerUI: {
      input:  { loader: "loader-input-building",  progress: "progress-input-building",  placeholder: "placeholder-input-building" },
      gt:     { loader: "loader-gt-building",     progress: "progress-gt-building",     placeholder: "placeholder-gt-building" },
      vecunico:  { loader: "loader-vecunico-building",  progress: "progress-vecunico-building",  placeholder: "placeholder-vecunico-building" },
      method: { loader: "loader-method-building", progress: "progress-method-building", placeholder: "placeholder-method-building" }
    }
  });

  if (buildingGroup) {
    GROUPS.push(buildingGroup);
    bindThumbnailsForGroup(buildingGroup, "thumbnail-building", "placeholder-qualitative");
    bindDeviationSliderForGroup(buildingGroup, "deviation-slider-building");
  }

  const abcGroup = initGroup({
    datasetDir: "abc",
    defaultMethod: "paco",
    useDeviation: true,
    alignToGround: true,
    upAxis: "z",
    doubleSided: true,
    autoRotateAxisLocal: "z",
    cameraFov: 35,
    autoFrame: true,
    autoFrameViewDir: [0.0, 0.5, 0.5],
    autoFrameTargetHeightFactor: 0.35,
    autoFrameDistanceMultiplier: 1.1,

    sampleContainerId: "thumbnail-abc",
    methodContainerId: "comparison-methods-abc",

    viewerIds: {
      input: "input-viewer-abc",
      gt: "gt-viewer-abc",
      vecunico: "vecunico-viewer-abc",
      method: "method-viewer-abc"
    },
    viewerUI: {
      input:  { loader: "loader-input-abc",  progress: "progress-input-abc",  placeholder: "placeholder-input-abc" },
      gt:     { loader: "loader-gt-abc",     progress: "progress-gt-abc",     placeholder: "placeholder-gt-abc" },
      vecunico:  { loader: "loader-vecunico-abc",  progress: "progress-vecunico-abc",  placeholder: "placeholder-vecunico-abc" },
      method: { loader: "loader-method-abc", progress: "progress-method-abc", placeholder: "placeholder-method-abc" }
    }
  });

  if (abcGroup) {
    GROUPS.push(abcGroup);
    bindThumbnailsForGroup(abcGroup, "thumbnail-abc", "placeholder-abc");
    bindDeviationSliderForGroup(abcGroup, "deviation-slider-abc");
  }


  const bimGroup = initGroup({
    datasetDir: "bim",
    defaultMethod: "paco",
    useDeviation: true,
    alignToGround: true,
    upAxis: "z",
    doubleSided: true,
    autoRotateAxisLocal: "z",
    cameraFov: 35,
    autoFrame: true,
    autoFrameViewDir: [0.0, 0.5, 0.5],
    autoFrameTargetHeightFactor: 0.35,
    autoFrameDistanceMultiplier: 1.1,
    primitiveCarouselIntervalMs: 1000,
    primitiveCarouselShuffle: true,
    sampleContainerId: "thumbnail-bim",
    // methodContainerId: "comparison-methods-bim",

    viewerIds: {
      input: "input-viewer-bim",
      gt: "gt-viewer-bim",
      vecunico: "vecunico-viewer-bim",
      primitive: "primitive-viewer-bim"
    },
    viewerUI: {
      input:  { loader: "loader-input-bim",  progress: "progress-input-bim",  placeholder: "placeholder-input-bim" },
      gt:     { loader: "loader-gt-bim",     progress: "progress-gt-bim",     placeholder: "placeholder-gt-bim" },
      vecunico:  { loader: "loader-vecunico-bim",  progress: "progress-vecunico-bim",  placeholder: "placeholder-vecunico-bim" },
      primitive: { loader: "loader-primitive-bim", progress: "progress-primitive-bim", placeholder: "placeholder-primitive-bim" }
    }
  });

  if (bimGroup) {
    GROUPS.push(bimGroup);
    bindThumbnailsForGroup(bimGroup, "thumbnail-bim", "placeholder-bim");
    bindDeviationSliderForGroup(bimGroup, "deviation-slider-bim");
  }

  if (GROUPS.length === 0) return;

  window.addEventListener("resize", () => {
    for (const g of GROUPS) handleResize(g.viewers);
  });

  RUNNING = true;
  animateGroups();
});



const getPrimitiveCarouselIds = (function () {
  const cache =  {
  "00005229": ["group_0", "group_1", "group_3", "group_4", "group_5", "group_6"],
  "bag_0518100000336675": ["group_0", "group_1","group_2", "group_3", "group_4", "group_5", "group_6", "group_7", "group_8", "group_9", "group_10", "group_12", "group_14", "group_15"]
};

  return function (group, sampleId) {
    if (!sampleId) return [];

    // ✅ 命中 cache
    if (cache[sampleId]) {
      return cache[sampleId].slice(); // 返回副本，防止外部修改
    }

    // ❗未命中 cache：从 PRIMITIVE_INDEX 生成
    return []
  };
})();


function getPrimitiveMeta(sampleId, primitiveId) {
  return window.PRIM_META?.[sampleId]?.[primitiveId] || null;
}


window.PRIM_META = {
  "00005229": {
    "group_0": { type: "plane", params: { a: 1.000, b: 0.014, c:  -0.009, d: 0.339} },
    "group_1": { type: "plane", params: { a: 1.000, b: 0.020, c:  -0.011, d: -0.324}  },
    "group_3": { type: "cylinder", params: { axis:[-1.000, -0.0126, 0.003], point: [-0.000,0.001, 0.001], radius: 0.081} },
    "group_4": { type: "cone", params: { axis: [0.825, -0.471, 0.311], apex: [-1.396, 0.760, -0.486], halfangle: 0.135 } },
    "group_5": { type: "cylinder", params: { axis:[0.998, -0.014, 0.055], point: [0.001, 0.018, -0.021], radius: 0.158} },
    "group_6": { type: "plane", params: { a: 1.000, b:  -0.004, c: -0.025, d: -0.151} },

  },
  "bag_0518100000336675": {
    "group_0":  { type: "plane", params: { a: -0.379, b:  0.555, c:  0.741, d: -0.091 } },
    "group_1":  { type: "plane", params: { a:  0.001, b:  0.019, c:  1.000, d:  0.009 } },
    "group_2":  { type: "plane", params: { a: -0.798, b: -0.602, c:  0.008, d: -0.048 } },
    "group_3":  { type: "plane", params: { a: -0.001, b: -0.034, c: -0.999, d: -0.060 } },
    "group_4":  { type: "plane", params: { a:  0.419, b: -0.534, c:  0.735, d: -0.112 } },
    "group_5":  { type: "plane", params: { a: -0.821, b: -0.570, c:  0.006, d: -0.046 } },
    "group_6":  { type: "plane", params: { a: -0.569, b:  0.822, c: -0.032, d: -0.125 } },
    "group_7":  { type: "plane", params: { a:  0.562, b: -0.826, c:  0.034, d: -0.127 } },
    "group_8":  { type: "plane", params: { a:  0.565, b: -0.825, c:  0.020, d:  0.001 } },
    "group_9":  { type: "plane", params: { a:  0.813, b:  0.582, c: -0.021, d: -0.066 } },
    "group_10": { type: "plane", params: { a: -0.811, b: -0.585, c:  0.015, d: -0.102 } },
    "group_12": { type: "plane", params: { a: -0.549, b: -0.367, c:  0.751, d: -0.065 } },
    "group_14": { type: "plane", params: { a:  0.824, b:  0.567, c: -0.003, d: -0.002 } },
    "group_15": { type: "plane", params: { a: -0.582, b:  0.812, c: -0.039, d: -0.082 } },
  
}};

// function formatPrimitiveMeta(meta) {
//   if (!meta) return "No semantics/params";
//   const type = meta.type ?? "unknown";
//   const params = meta.params ?? {};
//   // 简单格式化（你可以做得更漂亮）
//   return `${type}\n` + Object.entries(params)
//     .map(([k,v]) => `${k}: ${Array.isArray(v) ? JSON.stringify(v) : v}`)
//     .join("\n");
// }

function formatPrimitiveMeta(meta) {
  if (!meta) return "No semantics / params";

  const type = meta.type ?? "unknown";

  // 统一拿参数（兼容 params / pparams）
  const rawParams = meta.params ?? meta.pparams ?? {};

  const lines = [];
  lines.push(`type: ${type}`);

  for (const [key, value] of Object.entries(rawParams)) {
    lines.push(`  ${key}: ${formatValue(value)}`);
  }

  return lines.join("\n");
}

function formatValue(v) {
  if (v == null) return "null";

  // number
  if (typeof v === "number") {
    return v.toFixed(3);
  }

  // array: [x,y,z]
  if (Array.isArray(v)) {
    return `(${v.map(x => 
      typeof x === "number" ? x.toFixed(3) : x
    ).join(", ")})`;
  }

  // object: {x:?, y:?, z:?}
  if (typeof v === "object") {
    return `{ ${Object.entries(v)
      .map(([k,val]) =>
        `${k}: ${typeof val === "number" ? val.toFixed(3) : val}`
      )
      .join(", ")} }`;
  }

  return String(v);
}

function updatePrimitiveMetaPanel(viewer, sampleId, primitiveId) {
  if (!viewer?.metaEl) return;

  const meta = getPrimitiveMeta(sampleId, primitiveId);

  viewer.metaEl.textContent = meta
    ? formatPrimitiveMeta(meta)
    : "No primitive meta";

  viewer.metaEl.style.display = "block";
}



function buildPrimitiveFaceListPath(datasetDir, sampleId, method, group, primitive_id) {
  // 你这里 primitive_id 已经替代 frame 了，所以沿用同样的 base
  let frame = frameStrFromSliderValue(group?.deviationValue ?? 0);
  if (primitive_id != null) frame = primitive_id;

  const base = `./static/data/${datasetDir}/${sampleId}/${method}/${frame}`;
  // 约定：每个 primitive 对应一个 faces list JSON
  return `${base}_faces.json`;
}


// const _blueFaceCache = new Map(); // key: url -> Promise<number[]>

// async function getBlueFaceListForPrimitive(datasetDir, sampleId, method, group, primitive_id) {
//   const url = buildPrimitiveFaceListPath(datasetDir, sampleId, method, group, primitive_id);

//   if (_blueFaceCache.has(url)) return _blueFaceCache.get(url);

//   const p = (async () => {
//     const res = await fetch(url, { cache: "force-cache" });
//     if (!res.ok) {
//       // 没有 face list 就当成空（全灰）
//       return [];
//     }
//     const data = await res.json();

//     // 兼容两种格式：
//     // 1) [0,1,2,...]
//     // 2) { blueTris: [0,1,2,...] }
//     let blueTris = Array.isArray(data) ? data : (data?.blueTris ?? []);
//     if (!Array.isArray(blueTris)) blueTris = [];

//     // 保证是整数数组
//     return blueTris
//       .map(x => Number(x))
//       .filter(x => Number.isFinite(x))
//       .map(x => Math.floor(x));
//   })();

//   _blueFaceCache.set(url, p);
//   return p;
// }


const _blueFaceCache = new Map(); // url -> Promise<number[]>

async function getBlueFaceListForPrimitive(datasetDir, sampleId, method, group, primitive_id) {
  const url = buildPrimitiveFaceListPath(datasetDir, sampleId, method, group, primitive_id);

  if (_blueFaceCache.has(url)) return _blueFaceCache.get(url);

  const promise = (async () => {
    const res = await fetch(url, { cache: "force-cache" });
    // const res = await fetch(url, { cache: "reload" });
    if (!res.ok) return [];

    const data = await res.json();
    let blueTris = Array.isArray(data) ? data : (data?.blueTris ?? []);
    if (!Array.isArray(blueTris)) blueTris = [];

    return blueTris
      .map(Number)
      .filter(Number.isFinite)
      .map(Math.floor);
  })();

  _blueFaceCache.set(url, promise);
  return promise;
}