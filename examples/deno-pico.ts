import pico from './pico.js'
import { decode } from 'https://deno.land/x/jpegts@1.1/mod.ts'

const facefinder_bytes = await Deno.readFile('./examples/facefinder')
const facefinder_classify_region = pico.unpack_cascade(facefinder_bytes)

/**
 * a function to transform an RGBA image to grayscale
 */
function rgba_to_grayscale(rgba, nrows, ncols) {
  var gray = new Uint8Array(nrows * ncols)
  for (var r = 0; r < nrows; ++r)
    for (var c = 0; c < ncols; ++c)
      // gray = 0.2*red + 0.7*green + 0.1*blue
      gray[r * ncols + c] =
        (2 * rgba[r * 4 * ncols + 4 * c + 0] +
          7 * rgba[r * 4 * ncols + 4 * c + 1] +
          1 * rgba[r * 4 * ncols + 4 * c + 2]) /
        10
  return gray
}

async function find_face(image_filepath: string, stride?: number) {
  const raw = await Deno.readFile(image_filepath)
  const image_data = decode(raw)
  // const image_data = image_data_flat.reduce((acc, ))
  // console.log(image_data)
  const grey_image_data = rgba_to_grayscale(image_data, image_data.height, image_data.width)
  // console.log(image_data.height, image_data.width)
  const image = {
    pixels: rgba_to_grayscale(image_data, image_data.height, image_data.width),
    nrows: image_data.height,
    ncols: image_data.width,
    // ldim: image_data.width // ? TODO
    ldim: stride ?? image_data.width
  }
  const params = {
    shiftfactor: 0.01, // move the detection window by 10% of its size
    minsize: 20, // minimum size of a face (not suitable for real-time detection, set it to 100 in that case)
    maxsize: 1000, // maximum size of a face
    scalefactor: 0.9 // for multiscale processing: resize the detection window by 10% when moving to the higher scale
  }

  // run the cascade over the image
  // detections is an array that contains (r, c, s, q) quadruplets
  // (representing row, column, scale and detection score)
  let detections = pico.run_cascade(image, facefinder_classify_region, params)
  // cluster the obtained detections
  detections = pico.cluster_detections(detections, 0.2) // set IoU threshold to 0.2
  // draw results
  const qthresh = 5.0 // this constant is empirical: other cascades might require a different one

  // this draws the rectangles, it is not relevant to the face tracking
  const rectangles = detections.map(([x, y, w, h]) => `rectangle ${x},${y} ${w},${h}`).join(' ')
  const proc = Deno.run({
    cmd: [
      'convert',
      image_filepath,
      '-fill',
      'none',
      '-stroke',
      'red',
      '-draw',
      rectangles,
      'preview.jpg'
    ]
  })
  const result = await proc.status()
  if (!result.success) Deno.exit(1)
  console.log(image_filepath, `(${image_data.width}x${image_data.height})`, 'found', detections.length, 'faces with ldim', image.ldim)
  return detections
}

// 419 feels extremely arbitrary
await find_face('./samples/6627147.jpeg', 400)
await find_face('./samples/6627147.jpeg', 419)
await find_face('./samples/6627147.jpeg', 420)
await find_face('./samples/6627147.jpeg', 480)
