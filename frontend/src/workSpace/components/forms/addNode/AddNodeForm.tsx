import React, { FC, memo } from "react";
import { useFormik } from "formik";
import { useDispatch, useSelector } from "react-redux";
import scss from "./addNodeForm.module.scss";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Paper } from "@material-ui/core";

import AddNodeFormTitle from "./AddNodeFormTitle";
import AddNodeFormName from "./name/AddNodeFormName";
import { StateType } from "../../../../redux/store";
import AddNodeFormParents from "./parents/AddNodeFormParents";
import AddNodeFormChildren from "./children/AddNodeFormChildren";
import AddNodeFormType from "./type/AddNodeFormType";
import {
  actionsSandbox,
  editMainGraph,
} from "../../../../redux/reducers/sandBox/sandbox-reducer";
import ButtonsCancelAndSubmit from "../../../../ui/formik/buttons/ButtonsCancelAndSubmit";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      width: 278,

      position: "absolute",
      right: 0,

      zIndex: 3,
    },
    buttonsPosition: {
      marginTop: 16,
    },
  })
);

const creatingConnections = (
  id: number,
  childrens: number[],
  parents: number[]
) => {
  let childrensArr: { source: number; target: number }[] = [];
  let paretnsArr: { source: number; target: number }[] = [];

  if (childrens.length !== 0) {
    childrensArr = childrens.map((item) => {
      return { source: id, target: item };
    });
  }
  if (parents.length !== 0) {
    paretnsArr = parents.map((item) => {
      return { source: item, target: id };
    });
  }

  return [...childrensArr, ...paretnsArr];
};

interface I {
  setFromClosed(): void;
  type: "edit" | "new";
}

const AddNodeForm: FC<I> = ({ setFromClosed, type }) => {
  const classes = useStyles();
  const dispatch = useDispatch();
  const { mainGraph } = useSelector((state: StateType) => state.sandbox_Egor);
  const { sandbox_Initial_values_edit_piont } = useSelector(
    (state: StateType) => state.sandBox
  );

  const newInitialValues = {
    name: "",
    id: mainGraph.nodes.length + 1000,
    parents: "",
    children: "",
    type: "model", //должен быть select
  };

  const editInitialValues = {
    name: sandbox_Initial_values_edit_piont?.display_name as string,
    id: sandbox_Initial_values_edit_piont?.id as number,
    parents: (sandbox_Initial_values_edit_piont?.parents as number[]).join(),
    children: (sandbox_Initial_values_edit_piont?.children as number[]).join(),
    type: sandbox_Initial_values_edit_piont?.type as string, //должен быть select
  };

  const formik = useFormik({
    initialValues: type === "new" ? newInitialValues : editInitialValues,
    // validationSchema,
    onSubmit: (values) => {
      // console.log(`values`, values);
      // создаем массив стрелок

      if (type === "edit") {
        dispatch(
          editMainGraph(
            mainGraph,
            [
              {
                id: values.id,
                display_name: values.name,
                model_name: values.name,
                type: values.type,
                params: {},
                parents: JSON.parse(`[${values.parents}]`),
                children: JSON.parse(`[${values.children}]`),
              },
            ],
            creatingConnections(
              values.id,
              JSON.parse(`[${values.children}]`),
              JSON.parse(`[${values.parents}]`)
            )
          )
        );
      } else {
        creatingConnections(
          values.id,
          JSON.parse(`[${values.children}]`),
          JSON.parse(`[${values.parents}]`)
        );

        dispatch(
          actionsSandbox.addNodeMainGraph({
            nodes: [
              {
                id: values.id,
                display_name: values.name,
                model_name: values.name,
                type: values.type,
                params: {},
                parents: JSON.parse(`[${values.parents}]`),
                children: JSON.parse(`[${values.children}]`),
              },
            ],
            edges: creatingConnections(
              values.id,
              JSON.parse(`[${values.children}]`),
              JSON.parse(`[${values.parents}]`)
            ),
          })
        );
      }

      setFromClosed();
    },
  });

  return (
    <Paper elevation={3} className={classes.root}>
      <AddNodeFormTitle title={type === "edit" ? "edit Point" : "add Point"} />
      <div className={scss.formRoot}>
        <form onSubmit={formik.handleSubmit}>
          <AddNodeFormName
            values={formik.values}
            onChange={formik.handleChange}
          />
          <AddNodeFormParents
            values={formik.values}
            onChange={formik.handleChange}
          />
          <AddNodeFormChildren
            values={formik.values}
            onChange={formik.handleChange}
          />
          <AddNodeFormType
              values={formik.values}
              onChange={formik.handleChange}
          />
          <div className={classes.buttonsPosition}>
            <ButtonsCancelAndSubmit
              disabled={false}
              onCancelClick={setFromClosed}
            />
          </div>
        </form>
      </div>
    </Paper>
  );
};

export default memo(AddNodeForm);
