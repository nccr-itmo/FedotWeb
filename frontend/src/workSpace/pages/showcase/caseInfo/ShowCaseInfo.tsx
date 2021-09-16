import React, { memo } from "react";
import scss from "./showCaseInfo.module.scss";
import { useSelector } from "react-redux";

import { StateType } from "../../../../redux/store";
import PageTitle from "../../../components/pageTitle/PageTitle";
import ShowCaseInfoCard from "./card/ShowCaseInfoCard";

const ShowCaseInfo = () => {
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  return (
    <div className={scss.root}>
      <div className={scss.header}>
        <PageTitle title={show_case_by_id?.title as string} />
      </div>
      <ShowCaseInfoCard />
    </div>
  );
};

export default memo(ShowCaseInfo);
