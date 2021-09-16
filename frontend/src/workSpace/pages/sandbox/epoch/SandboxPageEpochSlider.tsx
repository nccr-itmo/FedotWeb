import React, { memo, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Slider } from "@material-ui/core";
import { withStyles } from "@material-ui/core/styles";
import {
  getEpochs,
  setSelectedEpoch,
} from "../../../../redux/reducers/sandBox/sandBoxReducer";
import { StateType } from "../../../../redux/store";
import { getMainGraph } from "../../../../redux/reducers/sandBox/sandbox-reducer";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      padding: "0 32px",
      width: "100%",

      display: "flex",
      flexDirection: "row",
      alignItems: "center",
    },
  })
);

const CustomSlider = withStyles({
  root: {
    // width: "558px",
    height: 8,
    zIndex: 2,

    color: "#263238",
  },
  thumb: {
    height: 24,
    width: 24,
    backgroundColor: "#fff",
    border: "2px solid currentColor",
    marginTop: -8,
    marginLeft: -12,
    "&:focus, &:hover, &$active": {
      boxShadow: "inherit",
    },
  },
  active: {},
  valueLabel: {
    left: "calc(-50% + 4px)",
  },
  track: {
    height: 8,
  },
  rail: {
    height: 8,
  },
  mark: {
    backgroundColor: "#fff",
    border: "1px solid #000",
    borderRadius: "50%",
    height: 8,
    width: 8,
    marginTop: -1,
  },
})(Slider);

const SandboxPageEpochSlider = () => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  const { sandBox_epochs } = useSelector((state: StateType) => state.sandBox);

  useEffect(() => {
    if (show_case_by_id) {
      dispatch(getEpochs(show_case_by_id.case_id));
    }
  }, [dispatch, show_case_by_id]);

  function valuetext(value: number) {
    return `${value}`;
  }

  const onChangeEpoch = (event: any, value: any) => {
    // console.log(`value`, value);
    // console.log(`sandBox_epochs`, sandBox_epochs);
    if (sandBox_epochs) {
      // sandBox_epochs.find((epoch) => epoch.epoch_num === value);
      dispatch(
        setSelectedEpoch(
          sandBox_epochs.find((epoch) => epoch.epoch_num === value)!
        )
      );
      dispatch(
        getMainGraph(
          sandBox_epochs.find((epoch) => epoch.epoch_num === value)!.pipeline_id
        )
      );
    }
  };

  return sandBox_epochs ? (
    <div className={classes.root}>
      <CustomSlider
        defaultValue={sandBox_epochs?.length}
        getAriaValueText={valuetext}
        onChange={(e, v) => onChangeEpoch(e, v)}
        aria-labelledby="discrete-slider"
        valueLabelDisplay="auto"
        step={1}
        marks
        min={1}
        max={sandBox_epochs?.length}
      />
    </div>
  ) : (
    <></>
  );
};

export default memo(SandboxPageEpochSlider);
