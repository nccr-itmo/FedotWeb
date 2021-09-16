import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
// import { useLocation } from "react-router-dom";

import { makeStyles } from "@material-ui/core/styles";

import { StateType } from "../../../../redux/store";
import { getMainGraph } from "../../../../redux/reducers/sandBox/sandbox-reducer";
import GraphEditorDirectedGraph from "./GraphEditorDirectedGraph/GraphEditorDirectedGraph";
import AddNode from "./AddNode";
import EditGrarhOnServer from "./EditGrarhOnServer/EditGrarhOnServer";

const useStyles = makeStyles(() => ({
  root: {
    padding: 32,
    height: 522,

    overflow: "auto",

    background: "#FFFFFF",
    borderRadius: 8,
  },
}));

const GraphEditor = () => {
  const classes = useStyles();
  // const { search } = useLocation();
  const dispatch = useDispatch();

  const { mainGraph } = useSelector((state: StateType) => state.sandbox_Egor);
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  useEffect(() => {
    if (show_case_by_id) {
      // const { uid } = deserializeQuery(search);
      //TODO:uid нужен реальный
      dispatch(getMainGraph(show_case_by_id.pipeline_id));
    }
  }, [dispatch, show_case_by_id]);

  // useEffect(() => {
  //   // console.log(`### mainGraph`, mainGraph);
  // }, [mainGraph]);

  return (
    <div className={classes.root}>
      <EditGrarhOnServer />
      <AddNode />
      {mainGraph && (
        <GraphEditorDirectedGraph
          edgesData={mainGraph.edges}
          nodesData={mainGraph.nodes}
        />
      )}
    </div>
  );
};
export default GraphEditor;
