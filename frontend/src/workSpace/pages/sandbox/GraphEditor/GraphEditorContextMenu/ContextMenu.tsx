import React, { FC } from "react";
import scss from "./contextMenu.module.scss";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Button, Paper } from "@material-ui/core";

import { offsetContextMenuType } from "../GraphEditorDirectedGraph/GraphEditorDirectedGraph";
import AddNodeFormTitle from "../../../../components/forms/addNode/AddNodeFormTitle";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      position: "absolute",
      top: 0,
      left: 0,

      display: "flex",
      flexDirection: "column",
    },
    submitButton: {
      marginTop: 4,
      padding: "0 8px",
      width: "100%",

      textTransform: "none",
      borderRadius: 4,
      background: "#828282",

        "&:disabled": {
          background: "#ECEFF1",
        },
        "&:hover": {
          background: "#515B5F",
        },
      },
  })
);

export interface IContextMenu {
  buttons: { name: string; action(): void }[];
  offset: offsetContextMenuType | undefined;
}

const ContextMenu: FC<IContextMenu> = ({ buttons, offset }) => {
  const classes = useStyles();

  let position: any;

  if (offset) {
    position = {
      position: "absolute",
      top: `${offset.y}px`,
      left: `${offset.x}px`,
      zIndex: 3,
    };
  }

  return (
    <Paper className={classes.root} style={position} elevation={3}>
      <AddNodeFormTitle title="point menu" />
      <div className={scss.buttonsContainer}>
        {buttons.map((item) => (
          <Button
            onClick={item.action}
            className={classes.submitButton}
            key={item.name}
          >
            <p className={scss.buttonTexxt}>{item.name}</p>
          </Button>
        ))}
      </div>
    </Paper>
  );
};
export default ContextMenu;
