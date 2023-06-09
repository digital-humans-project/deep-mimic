//
// Created by Dongho Kang on 10.09.22.
//

#include "pylocobase/sim/Simulator.h"

namespace pyloco {

Simulator::Simulator(double simTimeStepSize, double controlTimeStepSize, uint robotModel, bool loadVisuals)
    : simTimeStepSize(simTimeStepSize), controlTimeStepSize(controlTimeStepSize), robotModel(RobotInfo::Model(robotModel)) {
    // populate robot
    const auto &info = robotInfos[int(robotModel)];

    // create robot
    rbsFilePath_ = info.rbsFilePath;
    robot_ = std::make_shared<crl::loco::LeggedRobot>(rbsFilePath_.c_str(), info.rsFilePath.c_str(), loadVisuals);

    for (auto it : info.limbNames) {
        robot_->addLimb(it.first, it.second);
    }

    robot_->showSkeleton = true;

    // create backend simulator
    rbEngine_ = std::make_shared<crl::loco::ODERBEngine>();
    rbEngine_->loadRBsFromFile(PYLOCO_DATA_FOLDER "/environment/Ground.rbs");
    rbEngine_->addRobotToEngine(robot_);

    // add boxes for throwing
    setupBox(box0, "box0");
    setupBox(box1, "box1");
    setupBox(box2, "box2");
    rbEngine_->addRigidBodyToEngine(box0);
    rbEngine_->addRigidBodyToEngine(box1);
    rbEngine_->addRigidBodyToEngine(box2);

    // reset scene
    resetToDefaultRobotState();

    // populate properties
    crl::loco::GCRR gcrr(robot_);
    crl::dVector q;
    gcrr.getQ(q);
    maxJointAngle_ = crl::dVector(robot_->getJointCount());
    minJointAngle_ = crl::dVector(robot_->getJointCount());
    nominalJointAngle_ = crl::dVector(robot_->getJointCount());
    jointScaleFactors_ = crl::dVector(robot_->getJointCount());
    zeroJointAngle_ = crl::dVector::Zero(robot_->getJointCount());

    for (int i = 0; i < robot_->getJointCount(); i++) {
        maxJointAngle_[i] = robot_->getJoint(i)->maxAngle;
        minJointAngle_[i] = robot_->getJoint(i)->minAngle;
        nominalJointAngle_[i] = q[gcrr.getQIndexForJoint(i)];
        //jointScaleFactors_[i] = min(abs(nominalJointAngle_[i] - minJointAngle_[i]), abs(nominalJointAngle_[i] - maxJointAngle_[i]));
        jointScaleFactors_[i] = 1.0;
    }
    nominalBaseHeight_ = info.targetHeight;
    numJoints_ = robot_->getJointCount();
}

void Simulator::resetToDefaultRobotState() {
    const auto &info = robotInfos[int(robotModel)];
    if (info.rsFilePath.empty()) {
        // use default angles if rs file is not specified
        robot_->setDefaultState();
    } else {
        // use rs file to reset
        crl::loco::RobotState rs(*robot_);
        rs.readFromFile(info.rsFilePath.c_str());
        robot_->setState(rs);
    }
    robot_->setRootState(crl::P3D(0, nominalBaseHeight_, 0), crl::Quaternion::Identity());

    // set initial command
    for (int i = 0; i < robot_->getJointCount(); i++) {
        robot_->getJoint(i)->controlMode = crl::loco::RBJointControlMode::POSITION_MODE;
        robot_->getJoint(i)->motorKp = motorsKp;
        robot_->getJoint(i)->motorKd = motorsKd;
        robot_->getJoint(i)->maxTorque = motorMaxTorque;
        robot_->getJoint(i)->desiredControlPosition = robot_->getJoint(i)->getCurrentJointAngle();
        robot_->getJoint(i)->desiredControlSpeed = 0;
        robot_->getJoint(i)->desiredControlTorque = 0;
    }
}

void Simulator::reset() {
    // note. this is a pure virtual function, but it's good to provide some default implementation.
    // in the base class implement override function as follows:
    //
    // DerivedSimulator::reset() override {
    //
    //     ... (some additional reset logic required in the derived class)
    // }
    timeStamp_ = 0;
    throwCounter = 0;
    resetToDefaultRobotState();
    setupBox(box0, "box0");
    setupBox(box1, "box1");
    setupBox(box2, "box2");
}

void Simulator::reset(const crl::dVector &q, const crl::dVector &qDot, const double & t0) {
    reset();
    setTimeStamp(t0);
    crl::loco::GCRR GCRR(robot_);
    GCRR.setQ(q);
    GCRR.setQDot(qDot);
    GCRR.syncRobotStateWithGeneralizedCoordinates();
    for (int i = 0; i < robot_->getJointCount(); i++) {
        robot_->getJoint(i)->desiredControlPosition = robot_->getJoint(i)->getCurrentJointAngle();
    }
}

crl::dVector Simulator::getObservation(bool reduced) const {
    crl::dVector obs;
    if (reduced) {
        crl::dVector q = getQ();
        crl::dVector qDot = getQDot();
        crl::resize(obs, q.size() + qDot.size() - 3);
        crl::V3D bodyLinVel = robot_->getTrunk()->getLocalCoordinates(robot_->getTrunk()->getVelocityForPoint_local(crl::P3D()));
        crl::V3D bodyAngVel = robot_->getTrunk()->getLocalCoordinates(robot_->getTrunk()->getAngularVelocity());
        obs << q[1], q[4], q[5], q.segment(6, numJoints_), bodyLinVel, bodyAngVel, qDot.segment(6, numJoints_);
    } else {
        crl::dVector q = getQ();
        crl::dVector qDot = getQDot();
        crl::resize(obs, q.size() + qDot.size());
        obs << q, qDot;
    }
    return obs;
}

crl::dVector Simulator::getScaledAction(const crl::dVector &action) const {
    return action.cwiseProduct(jointScaleFactors_) + nominalJointAngle_;
}

//TODO (Double check usage: Fatemeh has setJointGains and setJointMaxTorque)
void Simulator::applyControlSignal(const crl::dVector &jointTarget) {
    for (int i = 0; i < robot_->getJointCount(); i++) {
        robot_->getJoint(i)->controlMode = crl::loco::RBJointControlMode::POSITION_MODE;
        robot_->getJoint(i)->motorKp = motorsKp;
        robot_->getJoint(i)->motorKd = motorsKd;
        robot_->getJoint(i)->maxTorque = motorMaxTorque;
        robot_->getJoint(i)->desiredControlPosition = jointTarget[i];
        robot_->getJoint(i)->desiredControlSpeed = 0;
        robot_->getJoint(i)->desiredControlTorque = 0;
    }
}

void Simulator::simulationStep() {
    rbEngine_->step(simTimeStepSize);
    timeStamp_ += simTimeStepSize;
}

void Simulator::setQAndQDot(const crl::dVector &q, const crl::dVector &qDot) {
    crl::loco::GCRR gcrr(robot_);
    gcrr.setQ(q);
    gcrr.setQDot(qDot);
    gcrr.syncRobotStateWithGeneralizedCoordinates();
}

crl::dVector Simulator::getQ() const {
    crl::loco::GCRR gcrr(robot_);
    crl::dVector q;
    gcrr.getQ(q);
    return q;
}

crl::dVector Simulator::getQDot() const {
    crl::loco::GCRR gcrr(robot_);
    crl::dVector dq;
    gcrr.getQDot(dq);
    return dq;
}

crl::dVector Simulator::getRootOrientation() const{
    crl::loco::RobotState state(*robot_);
    crl::P3D pos = state.getPosition();
    crl::Quaternion orientation = state.getOrientation();
    return crl::dVector(orientation.coeffs());
}

crl::dVector Simulator::getIkSolverQ(const crl::dVector &q_inital,
                                    const crl::dVector &lf_pos,
                                    const crl::dVector &rf_pos,
                                    const crl::dVector &lh_pos,
                                    const crl::dVector &rh_pos) const {
    crl::loco::GCRR gcrr(robot_);
    
    const crl::dVector eePosTarget[4]{lf_pos,rf_pos,lh_pos,rh_pos};
    crl::Quaternion R_desired[4];
    crl::dVector q = q_inital, q_backup;
    uint k, k_max = 50;
    const double tolerance = 0.015, alpha = 0.2;

    gcrr.getQ(q_backup); // store the initial q
    gcrr.setQ(q);
    gcrr.syncRobotStateWithGeneralizedCoordinates();
    // Only apply ik in these end-effector related joints
    std::vector<int> joints_ind{1,2,4,5,7,8,10,11,13,14,16,17,27,28,30,31,33,34,36,37,38,40,42,39,41,43};
    
    // set current orientation as desired orientation
    for (int i = 0; i < robot_->getLimbCount(); i++) 
        R_desired[i] = gcrr.getOrientationFor(robot_->getLimb(i)->eeRB);
    
    // Method 1: Weighted equal priority version

    // Weight matirx: weight importance for pos_x,y,z and roatation_x,y,z
    Eigen::DiagonalMatrix<double,6> W(0.1,1,0.1,0.5,0.5,0.5);
    for (k = 0; k < k_max; k++) 
    {
        crl::dVector deltaq(q.size() - 6);
        deltaq.setZero();

        crl::Matrix J_pos,J_ori,J_pusedo_inv;
        crl::P3D target, eePos, p_FK;
        std::shared_ptr<crl::loco::RB> rb;
        crl::V3D err_pos, err_angular;
        crl::dVector err(6,1);
        crl::Quaternion R_now;
        double max_err = 0;

        for (int i = 0; i < robot_->getLimbCount(); i++) {
            if(i == 1 || i == 3)
                continue;
            eePos = robot_->getLimb(i)->ee->endEffectorOffset;
            rb = (robot_->getLimb(i)->eeRB);
            target = crl::getP3D(eePosTarget[i]);

            // Position Jacobian 
            gcrr.compute_dpdq(eePos, rb, J_pos);
            J_pos = J_pos.block(0,6,3,q.size() - 6).eval();
            J_pos = J_pos(Eigen::all,joints_ind);

            // Angular Jacobian
            gcrr.compute_angular_jacobian(rb, J_ori);
            J_ori = J_ori.block(0,6,3,q.size() - 6).eval();
            J_ori = J_ori(Eigen::all,joints_ind);
            crl::Matrix J_total(J_pos.rows() + J_ori.rows(), J_pos.cols());

            // Stack two J into J_total
            J_total << J_pos, J_ori;
            J_pusedo_inv = (J_total.transpose()*W*J_total).ldlt().solve(J_total.transpose()*W);
            
            // Position error
            p_FK = robot_->getLimb(i)->getEEWorldPos();
            err_pos = (crl::V3D(p_FK,target));
            
            // Angular error, use axis-angle representation
            R_now = gcrr.getOrientationFor(rb);
            crl::Quaternion rotate = R_desired[i] * R_now.inverse();
            crl::AngleAxisd aa(rotate);
            err_angular = aa.axis().normalized();
            double angle = aa.angle();
            err_angular *= angle;
            
            err << err_pos, err_angular;
            deltaq(joints_ind) += J_pusedo_inv*err;
            
            if (max_err < err_pos.norm())
                max_err = err_pos.norm();
            
        } 
        if(max_err < tolerance) 
            break;
        q.tail(q.size() - 6) += alpha*deltaq;
        gcrr.setQ(q);
        gcrr.syncRobotStateWithGeneralizedCoordinates();
    }
    
    // std::cout << k <<" aver: "<<max_err<<std::endl;
    // reset current physic to the initial state
    gcrr.setQ(q_backup);
    gcrr.syncRobotStateWithGeneralizedCoordinates();
    return q;

    /*
    // prioritization version
    for (k = 0; k < k_max; k++) 
    {
        crl::dVector deltaq(q.size() - 6);
        deltaq.setZero();

        crl::Matrix J_pos,J_ori,J_pusedo_inv,Jy, Jx, Jz, Jxz_pusedo_inv;
        crl::P3D target, eePos, p_FK;
        std::shared_ptr<crl::loco::RB> rb;
        crl::V3D err_angular;
        crl::dVector err(4,1), err2(2,1);
        crl::Quaternion R_now;
        double err_x, err_y, err_z;
        double max_err = 0;

        for (int i = 0; i < robot_->getLimbCount(); i++) {

            eePos = robot_->getLimb(i)->ee->endEffectorOffset;
            rb = (robot_->getLimb(i)->eeRB);
            target = crl::getP3D(eePosTarget[i]);

            // Position Jacobian 
            gcrr.compute_dpdq(eePos, rb, J_pos);
            J_pos = J_pos.block(0,6,3,q.size() - 6).eval();
            Jx = J_pos.row(0);
            Jy = J_pos.row(1);
            Jz = J_pos.row(2);
            
            Jx = Jx(Eigen::all,joints_ind);
            Jy = Jy(Eigen::all,joints_ind);
            Jz = Jz(Eigen::all,joints_ind);
            
            // Angular Jacobian
            gcrr.compute_angular_jacobian(rb, J_ori);
            J_ori = J_ori.block(0,6,3,q.size() - 6).eval();
            J_ori = J_ori(Eigen::all,joints_ind);
            crl::Matrix J_total(Jy.rows() + J_ori.rows(), Jy.cols());

            // Task 1, high priority, y value and ori
            J_total << Jy, J_ori;
            J_pusedo_inv = (J_total.transpose()*J_total).ldlt().solve(J_total.transpose());
            
            // Task 2, low priority, x,z value
            crl::Matrix Jxz(Jx.rows() + Jz.rows(), Jx.cols());
            Jxz << Jx,Jz;

            crl::Matrix I(J_total.cols(), J_total.cols()); 
            I.setIdentity();
            crl::Matrix N1;
            N1 = I - J_pusedo_inv*J_total;
            
            crl::Matrix Temp = Jxz * N1;
            Jxz_pusedo_inv = (Temp.transpose()*Temp).ldlt().solve(Temp.transpose());
            
            // Position error
            p_FK = robot_->getLimb(i)->getEEWorldPos();
            err_y = (crl::V3D(p_FK,target))[1];
            err_x = (crl::V3D(p_FK,target))[0];
            err_z = (crl::V3D(p_FK,target))[2];

            // Angular error, use axis-angle representation
            R_now = gcrr.getOrientationFor(rb);
            crl::Quaternion rotate = R_desired[i] * R_now.inverse();
            crl::AngleAxisd aa(rotate);
            err_angular = aa.axis().normalized();
            double angle = aa.angle();
            err_angular *= angle;

            err << err_y, err_angular; // for task 1
            err2 << err_x, err_z;      // for task 2
            
            deltaq(joints_ind) += J_pusedo_inv*err + N1*Jxz_pusedo_inv*(err2 - Jxz*(J_pusedo_inv*err));
            
            if (max_err < err_y)
                max_err = err_y;
            
        } 
        if(max_err < tolerance) 
            break;
        q.tail(q.size() - 6) += alpha*deltaq;
        gcrr.setQ(q);
        gcrr.syncRobotStateWithGeneralizedCoordinates();
    }
    */
}


std::vector<crl::dVector> Simulator::getFkEEPosQ(const crl::dVector &q_raw) const {
    crl::loco::GCRR gcrr(robot_);
    std::vector<crl::dVector> eePosTarget;
    crl::dVector q_backup;
    gcrr.getQ(q_backup); // store the initial q
    gcrr.setQ(q_raw);
    gcrr.syncRobotStateWithGeneralizedCoordinates();
    crl::P3D  p_FK;

    for (int i = 0; i < robot_->getLimbCount(); i++) 
    {
        p_FK = robot_->getLimb(i)->getEEWorldPos();
        eePosTarget.push_back(crl::V3D(p_FK));
    } 
    gcrr.setQ(q_backup);
    gcrr.syncRobotStateWithGeneralizedCoordinates();
    return eePosTarget;
}

crl::Matrix Simulator::getFeetPosition() const {
    crl::Matrix ret(robot_->getLimbCount(), 3);
    for (int i = 0; i < robot_->getLimbCount(); i++) {
        crl::P3D eePos = robot_->getLimb(i)->getEEWorldPos();
        ret.row(i) << eePos.x, eePos.y, eePos.z;
    }
    return ret;
}

crl::Matrix Simulator::getFeetVelocity() const {
    crl::Matrix ret(robot_->getLimbCount(), 3);
    for (int i = 0; i < robot_->getLimbCount(); i++) {
        crl::V3D eeVel = robot_->getLimb(i)->getEEWorldVel();
        ret.row(i) << eeVel.x(), eeVel.y(), eeVel.z();
    }
    return ret;
}

void Simulator::drawImGui() {
    ImGui::Checkbox("Show meshes", &robot_->showMeshes);
    ImGui::InputDouble("Kp", &motorsKp, 10);
    ImGui::InputDouble("Kd", &motorsKd, 1);
    ImGui::InputDouble("Max torque", &motorMaxTorque, 1);

    ImGui::Text("Throw Box");
    static Eigen::Vector2d box_speed_limits{0.0, 5.0};
    ImGui::SliderScalar("Box speed", ImGuiDataType_Double, &box_speed, &box_speed_limits[0], &box_speed_limits[1]);
    ImGui::SameLine();
    if (ImGui::Button("Throw")) {
        crl::dVector initialPos = Eigen::Vector3d::Random() * 2;
        throwBox(throwCounter % 3, box_speed, initialPos);
        throwCounter++;
    }
}

crl::dVector Simulator::getMotorTorques() const {
    crl::dVector torques;
    torques.resize(robot_->getJointCount());
    for (int i = 0; i < robot_->getJointCount(); i++) {
        torques[i] = robot_->getJoint(i)->motorFeedback;
    }
    return torques;
}

crl::dVector Simulator::getAllLoopMotorTorques() const {
    return allLoopMotorTorques;
}

bool Simulator::isRobotCollapsed() const {
    // TODO: we have to support other primitives too!
    bool bad_contact = false;
    for (int i = 0; i < robot_->getRigidBodyCount(); i++) {
        const auto &rb = robot_->getRigidBody(i);
        if (!rb->rbProps.endEffectorPoints.empty())
            continue;  // if the body has the end effector
        for (const auto &col : rb->rbProps.collisionShapes) {
            if (const auto &cs = std::dynamic_pointer_cast<crl::loco::RRBCollisionSphere>(col)) {
                crl::P3D temp = rb->getWorldCoordinates(cs->localCoordinates);
                if (temp.y < cs->radius * 2)
                    bad_contact = true;
            }
        }
    }
    return bad_contact;
}

crl::dVector Simulator::getFeetContact() const {
    crl::dVector isContact(robot_->getLimbCount());
    for (int i = 0; i < robot_->getLimbCount(); i++) {
        const auto limb = robot_->getLimb(i);
        if (limb->getEEWorldPos().y < limb->eeRB->rbProps.endEffectorRadius) {
            isContact[i] = true;
        } else {
            isContact[i] = false;
        }
    }
    return isContact;
}

void Simulator::setupBox(const std::shared_ptr<crl::loco::RB> &dynamicBox, std::string name) {
    dynamicBox->rbProps.mass = 5.0;
    dynamicBox->rbProps.fixed = false;
    dynamicBox->name = name;
    dynamicBox->rbProps.collisionShapes.push_back(
        std::make_shared<crl::loco::RRBCollisionBox>(crl::P3D(0.0, 0.0, 0.0), crl::Quaternion(0.0, 0.0, 0.0, 1.0).normalized(), crl::V3D(0.1, 0.1, 0.1)));
    dynamicBox->setPosition(crl::P3D(10.0, 0.0, 10.0));
    dynamicBox->setVelocity(crl::V3D(0, 0, 0));
}

void Simulator::throwBox(int boxIdx, double strength, crl::dVector initialPos) {
    crl::P3D finalPos = crl::P3D(initialPos[0], initialPos[1], initialPos[2]) + robot_->getRoot()->getWorldCoordinates(crl::P3D());
    finalPos.y = std::max(finalPos.y + 1.5, 0.0);
    crl::V3D finalVel = crl::V3D(-strength * initialPos[0], -strength * initialPos[1], -strength * initialPos[2]);
    switch (boxIdx) {
        case 1:
            box1->setPosition(finalPos);
            box1->setVelocity(finalVel);
            break;
        case 2:
            box2->setPosition(finalPos);
            box2->setVelocity(finalVel);
            break;
        default:
            box0->setPosition(finalPos);
            box0->setVelocity(finalVel);
            break;
    }
}

}  // namespace pyloco