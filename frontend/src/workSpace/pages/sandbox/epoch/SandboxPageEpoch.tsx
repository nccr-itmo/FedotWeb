import React, { memo } from "react";
import scss from "./sandboxPageEpoch.module.scss";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import TimelineIcon from "@material-ui/icons/Timeline";
import SandboxPageEpochSlider from "./SandboxPageEpochSlider";

const useStyles = makeStyles(() =>
  createStyles({
    icon: {
      height: 24,
      width: 24,
    },
  })
);

const SandboxPageEpoch = () => {
  const classes = useStyles();

  return (
    <div className={scss.root}>
      <div className={scss.titleContainer}>
        <TimelineIcon className={classes.icon} />
        <p className={scss.titleText}>Epoch</p>
      </div>
      <SandboxPageEpochSlider />
    </div>
  );
};

export default memo(SandboxPageEpoch);
