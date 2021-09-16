import React, { memo } from "react";
import { useSelector } from "react-redux";
import scss from "./showCaseInfoCardStructure.module.scss";

import { StateType } from "../../../../../../redux/store";
import { ICase } from "../../../../../../API/showCase/showCasesInterface";

const ShowCaseInfoCardStructure = () => {
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  return (
    <div className={scss.root}>
      <p className={scss.title}>Model structure</p>
      <div className={scss.imgContainer}>
        <img
          src={(show_case_by_id as ICase).icon_path as string}
          alt={"Fedot"}
          className={scss.img}
        />
      </div>
    </div>
  );
};

export default memo(ShowCaseInfoCardStructure);
