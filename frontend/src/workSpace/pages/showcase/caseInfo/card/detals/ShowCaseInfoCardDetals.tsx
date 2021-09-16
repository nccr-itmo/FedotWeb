import React, { memo } from "react";
import { useSelector } from "react-redux";
import { StateType } from "../../../../../../redux/store";
import scss from "./showCaseInfoCardDetals.module.scss";

import ShowCaseInfoCardDetalsItem from "./ShowCaseInfoCardDetalsItem";

const ShowCaseInfoCardDetals = () => {
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  return (
    <div className={scss.root}>
      <p className={scss.title}>Model details</p>
      <div className={scss.data}>
        {Object.entries(show_case_by_id?.details as object).map(
          (metric, index) => (
            <ShowCaseInfoCardDetalsItem
              metric={metric[0]}
              value={metric[1]}
              key={index + "o"}
            />
          )
        )}
      </div>
    </div>
  );
};

export default memo(ShowCaseInfoCardDetals);
