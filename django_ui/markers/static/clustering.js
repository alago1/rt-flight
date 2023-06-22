// https://github.com/dcousens/haversine-distance/blob/main/index.js

function haversineDistance(a, b) {
    // const R = 6378137  // equatorial mean radius of Earth (in meters)
    const R = 6371008.8
    
    // hav(theta) = hav(bLat - aLat) + cos(aLat) * cos(bLat) * hav(bLon - aLon)
    function hav(x) {
        const sinHalf = Math.sin(x / 2)
        return sinHalf * sinHalf
    }

    function toRad(x) {
        return x * Math.PI / 180
    }

    const aLat = toRad(Array.isArray(a) ? a[1] : a.latitude ?? a.lat)
    const bLat = toRad(Array.isArray(b) ? b[1] : b.latitude ?? b.lat)
    const aLng = toRad(Array.isArray(a) ? a[0] : a.longitude ?? a.lng ?? a.lon)
    const bLng = toRad(Array.isArray(b) ? b[0] : b.longitude ?? b.lng ?? b.lon)
  
    const ht = hav(bLat - aLat) + Math.cos(aLat) * Math.cos(bLat) * hav(bLng - aLng)
    return 2 * R * Math.asin(Math.sqrt(ht))
}

function dbscan_clustering(minPts, eps) {
    const dbscan = new DBSCAN()

    // https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    // const metersPerRadian = 6371008.8;
    // console.log("DBSCAN eps:", eps_meters / metersPerRadian)
    return (data) => (
        dbscan.run(data, eps, minPts, haversineDistance)
    )
}
