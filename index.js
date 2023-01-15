async function readJSON (event) {
	let str = event.target.result;
	let json = JSON.parse(str);
    console.log('Loading keyNet3')

	const session = await ort.InferenceSession.create('https://github.com/Entenzahn/adl_ws22/releases/download/v0.1-beta/keyNet.onnx');

        /*j = await fetch("./tensor.json")
            .then(res => res.json())
            .then(json => {
              // Do whatever you want
              // console.log(json)
              return json;
            });*/
        console.log(json)

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const x_tensor = new ort.Tensor('float32', json.flat(), [1,1,192,646])
        console.log(x_tensor)

        // prepare feeds. use model input names as keys.
        const feeds = { 'x': x_tensor };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const pred = results.y_hat.data;
        const max_class = pred.indexOf(Math.max.apply(Math, pred));


        const keys = ['A major', 'A minor', 'Ab major', 'Ab minor', 'B major', 'B minor',
       'Bb major', 'Bb minor', 'C major', 'C minor', 'D major', 'D minor',
       'Db major', 'Db minor', 'E major', 'E minor', 'Eb major',
       'Eb minor', 'F major', 'F minor', 'G major', 'G minor', 'Gb major',
       'Gb minor']

        document.getElementById('predictions').innerHTML = 'Predicted Key: '+ keys[max_class];
}

async function predictKey(event) {
  try {

	    event.preventDefault();

	    if (!file.value.length) return;

        let reader = new FileReader();

        reader.onload = readJSON;

        reader.readAsText(file.files[0]);

    } catch (e) {
        document.getElementById('predictions').innerHTML = 'failed to inference ONNX model:'+e;
    }

}