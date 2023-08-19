class HDBSCAN {
  constructor(minPts) {
    this.minPts = minPts;
    this.loaded = false;
    this.init();
  }

  async init() {
    const pyodide = await loadPyodide();
    const minPts = this.minPts;
    await pyodide.loadPackage(["scikit-learn", "numpy"]).then(() => {
      pyodide.runPython(
        `
              import numpy as np
              from sklearn.cluster import HDBSCAN

              def harv(a, b):
                lon1, lat1, r1 = a
                lon2, lat2, r2 = b
                R = 6371
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))

                return c*R - r1 - r2
  
              def hdbscan_cluster(X):
                X = np.array(X)
                hdb = HDBSCAN(min_cluster_size=${minPts}, metric='haversine', cluster_selection_method='leaf')
                return hdb.fit_predict(X)`
      );
    });
    this.pyodide = pyodide;
    this.clustering = pyodide.globals.get("hdbscan_cluster");
    this.loaded = true;
  }

  toRad(x) {
    return (x * Math.PI) / 180;
  }

  run(data) {
    if (!this.loaded) return [];
    if (data.length < this.minPts || data.length < 1) return [];

    data = data.map((x) => [this.toRad(x[0]), this.toRad(x[1])]);
    const labels = this.clustering(data).toJs();

    const num_clusters = Math.max(...labels) + 1;
    const clusters = new Array(num_clusters);

    labels.forEach((label, i) => {
      if (label < 0) return;
      if (!clusters[label]) clusters[label] = new Array();
      clusters[label].push(i);
    });

    return clusters;
  }
}
