import React, { FC } from "react";
import scss from "./addNodeForm.module.scss";

interface I {
  title: string;
}

const AddNodeFormTitle: FC<I> = ({ title }) => {
  return (
    <div className={scss.titleRoot}>
      <p className={scss.titleText}>{title}</p>
    </div>
  );
};

export default AddNodeFormTitle;
