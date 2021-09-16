import React, { useCallback, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
// import { useLocation } from "react-router-dom";

import { Grid } from "@material-ui/core";
import { makeStyles, createStyles } from "@material-ui/core/styles";

// import { StateType } from "../../store/store";
// import { deserializeQuery } from "../../../data/utils/query-helpers";
import HistoryModal from "./HistoryGraph/HistoryModal/HistoryModal";
import HistoryGraph from "./HistoryGraph/HistoryGraph";
import { getHistoryGraph } from "../../../redux/reducers/sandBox/sandbox-reducer";
import { StateType } from "../../../redux/store";
import HistoryPageChart from "./charts/HistoryPageChart";

const useStyles = makeStyles(() =>
  createStyles({
    item: {
      background: "#FFFFFF",
      borderRadius: 8,
    },
  })
);

const HistoryPage = () => {
  const classes = useStyles();

  const [uidModal, setUidModal] = useState("");
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  // const { search } = useLocation();
  const dispatch = useDispatch();
  const nodeHoverTooltip = useCallback((node) => {
    return `<div>     
        <p >${node}</p>
      </div>`;
  }, []);

  useEffect(() => {
    if (show_case_by_id) {
      dispatch(getHistoryGraph(show_case_by_id.case_id));
    }
  }, [dispatch, show_case_by_id]);

  const { historyGraph } = useSelector(
    (state: StateType) => state.sandbox_Egor
  );

  return (
    <Grid container spacing={0}>
      <Grid item xs={8} className={classes.item}>
        <HistoryGraph
          edgesData={historyGraph.edges}
          nodesData={historyGraph.nodes}
          onClick={(v) => {
            setUidModal(v);
          }}
          nodeHoverTooltip={nodeHoverTooltip}
        />
        {uidModal && (
          <HistoryModal
            uid={uidModal}
            handleClose={() => {
              setUidModal("");
            }}
          />
        )}
      </Grid>
      <Grid item xs={4}>
        <HistoryPageChart />
      </Grid>
    </Grid>
  );
};
// };

export default HistoryPage;
