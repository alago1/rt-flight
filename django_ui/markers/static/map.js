const copy = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors";
const url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const baseLayer = L.tileLayer(url, { attribution: copy });
const heatmapLayer = new HeatmapOverlay({
    useLocalExtrema: false,
    latField: "lat",
    lngField: "lng",
})
const map = L.map("map", { layers: [baseLayer, heatmapLayer] });
map.attributionControl.setPosition('bottomleft');
// map.locate()
// .on("locationfound", (e) => map.setView(e.latlng, 8))
// .on("locationerror", (e) => map.setView([0, 0], 5));

map.fitWorld();


// DBSCAN clustering setup
const dbscan = dbscan_clustering(3, 50);

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


async function load_markers(bbox_filter=null) {
    const queryParams = new URLSearchParams({ in_bbox: bbox_filter }).toString();
    const markers_url = '/api/markers/' + (bbox_filter !== null ? `?${queryParams}` : '');
    const response = await fetch(markers_url);
    const geojson = await response.json();
    return geojson;
}


async function render_markers({skip_on_no_new=false}) {
    const metersPerPixel = 40075016.686 * Math.abs(Math.cos(map.getCenter().lat * Math.PI/180)) / Math.pow(2, map.getZoom()+8);
    const scale = 5e-1;  // increase radius to make clusters more visible
    console.log(`Meters per pixel: ${metersPerPixel}`)

    const max_radius = 1000 / metersPerPixel; // 1km in pixels
    const min_radius = 10;  // pixels

    const markers = await load_markers(map.getBounds().toBBoxString());

    // if all markers are already known, don't re-render
    if (skip_on_no_new && markers.features.every(f => knownIds.has(f.properties.id))) {
        return;
    }
    
    const clusters = dbscan(markers.features.map(f => f.geometry.coordinates));

    const cluster_labels = new Array(markers.features.length).fill(-1);
    clusters.forEach((cluster, i) => cluster.forEach((idx) => cluster_labels[idx] = i));

    // add it as a feature so that it can be used in the popup
    markers.features.forEach((feature, i) => feature.properties.cluster = cluster_labels[i]);

    const heatmap_data = markers.features.map((feature) => ({
        lat: feature.geometry.coordinates[1],
        lng: feature.geometry.coordinates[0],
        radius: Math.max(Math.min(feature.properties.radius / metersPerPixel * scale, max_radius), min_radius),
    }));

    
    if (markerLayer !== null) {
        // find and remove only markers that are not in the map bounding box
        const new_markers_ids = new Set(markers.features.map(f => f.properties.id))
        const removable_layer_keys = Object.keys(markerLayer._layers)
                                    .filter(k => !new_markers_ids.has(markerLayer._layers[k].feature.id));
        removable_layer_keys.forEach(key => markerLayer._layers[key].removeFrom(map));
    }

    // find markers that are not in the layer (new markers) and add them to the map
    const existing_markers_ids = new Set(Object.keys(markerLayer?._layers ?? {}).map(key => markerLayer._layers[key].feature.id));
    const new_markers = markers.features.filter(f => !existing_markers_ids.has(f.properties.id));

    markerLayer = L.geoJSON(new_markers)
        .bindPopup(markerPopup)
        .addTo(map);

    const dom_icons = Object.keys(markerLayer._layers).map(key => (
        {
            id: markerLayer._layers[key].feature.id,
            icon: markerLayer._layers[key]._icon,
            cluster: markerLayer._layers[key].feature.properties.cluster,
        }
    )).sort((a, b) => a.id - b.id);

    
    // 4 color theorem implies that 4 colors are enough to color any map
    // but i haven't found a good (and efficient) way to apply it here so
    // i'm just using n random colors for n clusters
    const cluster_colors = nRandomColorFilters(clusters.length);

    dom_icons.forEach(({id, icon, cluster}) => {
        // add class property to dom icon
        icon.classList.add(`cluster-${cluster}`);
        prev_style = icon.getAttribute('style')
        icon.setAttribute('style', `${prev_style}; filter: ${cluster_colors[cluster]}`)
        icon.classList.add(`marker-${id}`)
        
        if (!knownIds.has(id)) {
            knownIds.add(id);
            icon.classList.add("new-marker");
            createRipple(icon);
            console.log()
        }
    })
    
    if(heatmap_data.length > 0)
        heatmapLayer.setData({ data: heatmap_data });
}



// Table logic
let table = null;
let showTable = null;

function toggleTable() {
    const table = document.getElementById("marker-table-wrapper");
    const toggleButton = document.getElementById("table-button");
    showTable = !showTable;  // will set to true on first call (!null === true)
    table.style.transform = `translateX(${showTable ? "-100" : "0"}%)`;
    toggleButton.style.rotate = `${showTable ? "180" : "0"}deg`;
}


async function update_table() {
    const markers = await load_markers();
    const data = markers.features.map((feature) => ({
        ...feature,
        properties: {
            ...feature.properties,
            radius: feature.properties.radius.toFixed(2),
            latitude: feature.geometry.coordinates[1].toFixed(8),
            longitude: feature.geometry.coordinates[0].toFixed(8),
        }
    }));

    if (table !== null) {
        const tableIds = new Set(table.getData().map(t => t.id));
        const new_data = data.filter(t => !tableIds.has(t.id));

        if (new_data.length !== 0) {
            await table.addData(new_data);
            table.setSort("id", "desc");
        }
        
        return;
    }

    table = new Tabulator("#marker-table", {
        data: data,
        layout: "fitColumns",
        pagination: "local",
        paginationSize: 35,
        paginationSizeSelector: [15, 35, 50, 100],
        paginationCounter: "rows",
        columns: [
            { title: "ID", field: "id", sorter: "number", headerFilter: true },
            { title: "Name", field: "properties.name", headerFilter: true },
            { title: "Confidence", field: "properties.confidence", sorter: "number", headerFilter: true },
            { title: "Radius", field: "properties.radius", sorter: "number", headerFilter: true },
            { title: "Latitude", field: "properties.latitude", headerFilter: true, headerSort: false },
            { title: "Longitude", field: "properties.longitude", headerFilter: true, headerSort: false },
        ],
        initialSort: [
            { column: "id", dir: "desc" },
        ],
    });

    // zoom to latest data if available
    if (data.length > 0) {
        const row = data.sort((a, b) => b.id - a.id)[0];
        map.setView([row.properties.latitude, row.properties.longitude], 10);
    }

    table.on("rowClick", async function(e, row) {
        const marker_pos = {
            lat: row._row.data.properties.latitude,
            lng: row._row.data.properties.longitude,
        };
        const center = map.getCenter()

        const showMarkerPopup = () => {
            const elems = document.getElementsByClassName(`marker-${row._row.data.id}`)
            console.log(elems)
            if (elems.length !== 0) {
                elems[0].click();
                createRipple(elems[0]);
                return;
            }
            setTimeout(showMarkerPopup, 500);
        }

        if (center.lat !== marker_pos.lat || center.lng !== marker_pos.lng) {
            map.setView(marker_pos, Math.max(map.getZoom(), 12), {animate: true});
            setTimeout(showMarkerPopup, 200);
        }
    });
}

toggleTable(); // open table on page load
update_table(); // populate table on page load



map.on("moveend", render_markers);
setInterval(() => render_markers({skip_on_no_new: true}), 5000);
setInterval(update_table, 5000)
