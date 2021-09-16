import React, { FC, memo } from "react";
import scss from "./showCaseInfoCardDetals.module.scss";

import { Grid, Tooltip } from "@material-ui/core";

interface IShowCaseInfoCardDetalsItem {
  metric: string;
  value: string;
}

const ShowCaseInfoCardDetalsItem: FC<IShowCaseInfoCardDetalsItem> = ({
  metric,
  value,
}) => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={8}>
        <Tooltip title={metric} placement="top-start">
          <p className={scss.metric}>{metric}</p>
        </Tooltip>
      </Grid>
      <Grid item xs={4}>
        <Tooltip title={value} placement="top">
          <p className={scss.value}>{value}</p>
        </Tooltip>
      </Grid>
    </Grid>
  );
};

export default memo(ShowCaseInfoCardDetalsItem);
