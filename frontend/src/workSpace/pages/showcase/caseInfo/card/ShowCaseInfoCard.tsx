import React, { memo } from "react";
import scss from "./showCaseInfoCard.module.scss";
import { useSelector } from "react-redux";

import { Grid } from "@material-ui/core";

import ShowCaseInfoCardStructure from "./structure/ShowCaseInfoCardStructure";
import ShowCaseInfoCardDetals from "./detals/ShowCaseInfoCardDetals";
import { StateType } from "../../../../../redux/store";
import ShowCaseInfoCardEditBtn from "./editButton/ShowCaseInfoCardEditBtn";

const ShowCaseInfoCard = () => {
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  return (
    <div className={scss.root}>
      <Grid container spacing={2}>
        <Grid item xs={9}>
          <ShowCaseInfoCardStructure />
        </Grid>
        <Grid item xs={3}>
          <ShowCaseInfoCardDetals />
        </Grid>
        <Grid item xs={12}>
          <p className={scss.description}>{show_case_by_id?.description}</p>
        </Grid>
        <ShowCaseInfoCardEditBtn />
      </Grid>
    </div>
  );
};

export default memo(ShowCaseInfoCard);
