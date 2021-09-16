import React, { memo } from "react";
import { useSelector } from "react-redux";

import { Fade, Grid } from "@material-ui/core";

import Cases from "./cases/Cases";
import ShowCaseInfo from "./caseInfo/ShowCaseInfo";
import { StateType } from "../../../redux/store";

const ShowcasePage = () => {
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  return show_case_by_id ? (
    <Grid container spacing={2}>
      <Grid item xs={8}>
        <Fade
          in={show_case_by_id !== null}
          mountOnEnter
          unmountOnExit
          timeout={1000}
        >
          <div>
            <ShowCaseInfo />
          </div>
        </Fade>
      </Grid>
      <Grid item xs={4}>
        <Fade
          in={show_case_by_id !== null}
          mountOnEnter
          unmountOnExit
          timeout={2000}
        >
          <div>
            <Cases />
          </div>
        </Fade>
      </Grid>
    </Grid>
  ) : (
    <Cases />
  );
};

export default memo(ShowcasePage);
