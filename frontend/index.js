import { BASE_URL } from './settings.js'
import { makeOptions, handleHttpErrors } from './utils.js'
import * as Plotly from 'https://cdn.plot.ly/plotly-2.16.1.min.js' 


async function fetchGet(endpoint){
    try {
        let res = await fetch(endpoint, makeOptions("GET"))
        return await handleHttpErrors(res)
    } catch (err) {
        console.error((err.message))
        if (err.apiError) {
            console.error("Api error: ", err.apiError)
        }
    }
    
}

function makeTable(element_id, dataset){
    var values = []

    for (var i = 0; i < Object.keys(dataset).length; i++){
        values.push(Object.values(dataset[i]))
    }
  
    var data = [{
        type: 'table',
        header: {
            values: [Object.keys(dataset)[0]],
            align: "center",
            line: {width: 1, color: 'black'},
            fill: {color: "grey"},
            font: {family: "Arial", size: 12, color: "white"}
        },
        cells: {
            values: values,
            align: "center",
            line: {color: "black", width: 1},
            font: {family: "Arial", size: 11, color: ["black"]}
        }
    }]
  
    Plotly.newPlot(element_id, data);
}

var dataset = await fetchGet(BASE_URL)
makeTable("content", dataset.root_data)


