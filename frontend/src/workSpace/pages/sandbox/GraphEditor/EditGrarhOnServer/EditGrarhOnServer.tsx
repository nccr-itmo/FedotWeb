import React, { memo } from "react";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";

import AddButton from "../../../../../ui/buttons/AddButton";
import { sendNewGraphOnServer } from "../../../../../redux/reducers/sandBox/sandBoxReducer";
import { StateType } from "../../../../../redux/store";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      position: "relative",
    },
    button: {
      position: "absolute",
      top: -20,
      left: 0,
      zIndex: 2,
    },
  })
);

const EditGrarhOnServer = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { mainGraph } = useSelector((state: StateType) => state.sandbox_Egor);
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  const sendNewDataOnServer = () => {
    if (show_case_by_id)
      dispatch(
        sendNewGraphOnServer(
          show_case_by_id.case_id,
          show_case_by_id.pipeline_id,
          mainGraph.nodes,
          mainGraph.edges
        )
      );
  };

  return (
    <div className={classes.root}>
      <div className={classes.button}>
        <AddButton onClick={sendNewDataOnServer} type="server">
          Evaluate
        </AddButton>
      </div>
    </div>
  );
};

export default memo(EditGrarhOnServer);
