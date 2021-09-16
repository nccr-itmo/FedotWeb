import React, { FC, useEffect } from "react";
import { Button, Dialog, Slide, Typography } from "@material-ui/core";
import { useDispatch, useSelector } from "react-redux";
import {
  createStyles,
  makeStyles,
  WithStyles,
  withStyles,
} from "@material-ui/core/styles";
import { TransitionProps } from "@material-ui/core/transitions";
import MuiDialogTitle from "@material-ui/core/DialogTitle";
import IconButton from "@material-ui/core/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import StarRateIcon from "@material-ui/icons/StarRate";
import { dataContextType } from "../GraphEditorDirectedGraph/GraphEditorDirectedGraph";
import { StateType } from "../../../../../redux/store";
import { actionsSandbox } from "../../../../../redux/reducers/sandBox/sandbox-reducer";
import { getParams } from "../../../../../redux/reducers/sandBox/sandBoxReducer";

const styles = () =>
  createStyles({
    root: {
      background: "#B0BEC5",
      margin: 0,
      display: "flex",
      justifyContent: "space-between",
      padding: "6px 12px",
    },
    closeButton: {
      padding: 2,
    },
  });
export interface DialogTitleProps extends WithStyles<typeof styles> {
  id: string;
  children: React.ReactNode;
  onClose: () => void;
}

const DialogTitle = withStyles(styles)((props: DialogTitleProps) => {
  const { children, classes, onClose, ...other } = props;

  return (
    <MuiDialogTitle disableTypography className={classes.root} {...other}>
      <Typography variant="subtitle1">{children}</Typography>
      {onClose ? (
        <IconButton
          aria-label="close"
          className={classes.closeButton}
          onClick={onClose}
        >
          <CloseIcon />
        </IconButton>
      ) : null}
    </MuiDialogTitle>
  );
});

const useStyles = makeStyles((theme) => ({
  dialog: {
    position: "absolute",
    top: "40%",
    right: "50px",
    transform: "translate(0, -50%)",
    width: 278,
  },
  content: {
    padding: "6px 12px",
  },
  line: {
    display: "flex",
  },
}));

const Transition = React.forwardRef(function Transition(
  props: TransitionProps & { children?: React.ReactElement<any, any> },
  ref: React.Ref<unknown>
) {
  return <Slide direction="left" ref={ref} {...props} />;
});

interface IGraphEditorModal {
  dataContext: dataContextType | undefined;
}

const GraphEditorModal: FC<IGraphEditorModal> = ({ dataContext }) => {
  const classes = useStyles();
  const dispatch = useDispatch();

  const { isOpenEditModal } = useSelector(
    (state: StateType) => state.sandbox_Egor
  );
  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  const handleClose = () => {
    dispatch(actionsSandbox.toggleEditModal());
  };

  useEffect(() => {
    if (show_case_by_id) {
      dispatch(getParams(show_case_by_id?.case_id));
    }
  }, [dispatch, show_case_by_id]);

  return (
    <Dialog
      hideBackdrop
      open={isOpenEditModal}
      TransitionComponent={Transition}
      onClose={handleClose}
      aria-labelledby="edit-modal-title"
      aria-describedby="edit-modal-description"
      classes={{
        paper: classes.dialog,
      }}
    >
      <DialogTitle id="history-dialog-title" onClose={handleClose}>
        XGB | ML model
      </DialogTitle>
      <div className={classes.content}>
        <div className={classes.line}>
          <StarRateIcon />
          <Typography>Rating 10/10</Typography>
        </div>
        <div className={classes.line}>
          <Typography>modal_name: </Typography>
          <Typography
            style={{ backgroundColor: "#F4F6F8", marginLeft: 10, padding: 2 }}
          >
            modal_name
          </Typography>
        </div>
        <Typography variant={"body1"}>Hyperparameters:</Typography>
        <div
          style={{
            backgroundColor: "#F4F6F8",
          }}
        >
          <Typography style={{ padding: "2px 4px" }} variant={"body2"}>
            Hyperparameter
          </Typography>
          <Typography style={{ padding: "2px 4px" }} variant={"body2"}>
            Hyperparameter
          </Typography>
          <Typography style={{ padding: "2px 4px" }} variant={"body2"}>
            Hyperparameter
          </Typography>
          <Typography style={{ padding: "2px 4px" }} variant={"body2"}>
            Hyperparameter
          </Typography>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-around",
          marginTop: 20,
          marginBottom: 20,
        }}
      >
        <Button onClick={handleClose} size={"small"}>
          cancel
        </Button>
        <Button
          onClick={() => {
            console.log(`### dataContext`, dataContext);
            if (dataContext) {
              dispatch(actionsSandbox.deleteNodeMainGraph(dataContext.data));
            }
          }}
          size={"small"}
          variant={"contained"}
        >
          delete
        </Button>
      </div>
    </Dialog>
  );
};
export default GraphEditorModal;
