const copy = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors";
const url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const baseLayer = L.tileLayer(url, { attribution: copy });
const heatmapLayer = new HeatmapOverlay({
    useLocalExtrema: false,
    latField: "lat",
    lngField: "lng",
})
const map = L.map("map", { layers: [baseLayer, heatmapLayer] });
map.locate()
.on("locationfound", (e) => map.setView(e.latlng, 8))
.on("locationerror", (e) => map.setView([0, 0], 5));

map.fitWorld();


// DBSCAN clustering setup
const dbscan = dbscan_clustering(3, 2000);

// UI
function markerPopup(layer) {
    return (
        `
        <div class="marker-popup">
            <span>ID ${layer.feature.id}: ${layer.feature.properties.name}</span>
            <span>Coordinates: ${layer.feature.geometry.coordinates[1].toFixed(4)}, ${layer.feature.geometry.coordinates[0].toFixed(4)}</span>
            <span>Confidence: ${layer.feature.properties.confidence}%</span>
            <span>Radius: ${layer.feature.properties.radius.toFixed(2)} meters</span>
            <span>Cluster: ${layer.feature.properties.cluster}</span>
        <div>
        `
    )
}

// marker rippler animation logic
function createRipple(markerIcon) {
    const circle = document.createElement("span");
    const radius = 20; // pixels
    circle.style.width = circle.style.height = `${radius}px`;
    const {x, y} = Array.from(CSSStyleValue.parse('transform', markerIcon.style.transform))
                                .filter(s => s instanceof CSSTranslate)[0]
    circle.style.left = `calc(${x.value}${x.unit} - ${radius / 2}px)`;
    circle.style.top = `calc(${y.value}${y.unit} - ${radius / 2}px)`;
    circle.classList.add('ripple');

    markerIcon.parentNode.appendChild(circle);
    setTimeout(() => circle.remove(), 5000);
}

const knownIds = new Set();
let markerLayer = null;


async function load_markers() {
    const markers_url = `/api/markers/?in_bbox=${
        map.getBounds().toBBoxString()
    }`;
    const response = await fetch(markers_url);
    const geojson = await response.json();
    return geojson;
}


async function render_markers() {
    const metersPerPixel = 40075016.686 * Math.abs(Math.cos(map.getCenter().lat * Math.PI/180)) / Math.pow(2, map.getZoom()+8);
    const scale = 1e3;  // increase radius to make clusters more visible
    console.log(`Meters per pixel: ${metersPerPixel}`)

    const max_radius = 1000 / metersPerPixel; // 1km in pixels
    const min_radius = 10;  // pixels

    const markers = await load_markers();
    
    const clusters = dbscan(markers.features.map(f => f.geometry.coordinates));

    const cluster_labels = new Array(markers.features.length).fill(-1);
    clusters.forEach((cluster, i) => cluster.forEach((idx) => cluster_labels[idx] = i));

    // add it as a feature so that it can be used in the popup
    markers.features.forEach((feature, i) => feature.properties.cluster = cluster_labels[i]);

    const heatmap_data = markers.features.map((feature) => ({
        lat: feature.geometry.coordinates[1],
        lng: feature.geometry.coordinates[0],
        radius: Math.max(Math.min(feature.properties.radius / metersPerPixel * scale, max_radius), min_radius),
        opacity: feature.properties.confidence / 100.,
    }));

    if (markerLayer !== null)
        markerLayer.removeFrom(map);

    markerLayer = L.geoJSON(markers)
        .bindPopup(markerPopup)
        .addTo(map);

    const dom_icons = Object.keys(markerLayer._layers).map(key => (
        {
            id: markerLayer._layers[key].feature.id,
            icon: markerLayer._layers[key]._icon,
            cluster: markerLayer._layers[key].feature.properties.cluster,
        }
    )).sort((a, b) => a.id - b.id);

    
    const cluster_colors = nRandomColorFilters(clusters.length);

    dom_icons.forEach(({id, icon, cluster}) => {
        // add class property to dom icon
        icon.classList.add(`cluster-${cluster}`);
        prev_style = icon.getAttribute('style')
        icon.setAttribute('style', `${prev_style}; filter: ${cluster_colors[cluster]}`)
        
        if (!knownIds.has(id)) {
            knownIds.add(id);
            icon.classList.add("new-marker");
            createRipple(icon);
        }
    })
    
    if(heatmap_data.length > 0)
        heatmapLayer.setData({ data: heatmap_data });
}

map.on("moveend", render_markers);
setInterval(render_markers, 5000);
