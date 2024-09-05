import {useNavigate} from 'react-router-dom'

const Navbar = () => {
  const navigate=useNavigate()

  return (
    <nav className="m-0 sticky bg-black !text-white w-full flex flex-row py-2.5 px-4 box-border items-center justify-between top-[0] z-[1] !font-manrope">
      <button className="cursor-pointer [border:none] p-0 bg-[transparent] flex flex-row items-center justify-start text-white text-xl font-manrope">
      Ai Tube
      </button>

      <div className=" flex flex-row items-center justify-start gap-[51px]">
        <div className="flex flex-row items-center justify-center">
          <button 
          onClick={()=>{navigate('/example')}}
          className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-xl leading-[20px] font-medium font-roboto text-white text-left inline-block">
            Home
          </button>
        </div>
        <div className="flex flex-row items-center justify-center">
          <button 
          onClick={()=>{navigate('/leader-board')}}
          className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-xl leading-[20px] font-medium font-roboto text-white text-left inline-block">
            LeaderBoard
          </button>
        </div>
        <div className="flex flex-row items-center justify-start">
          <button 
          onClick={()=>{navigate('/ai-code')}}
          className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-xl leading-[20px] font-medium font-roboto text-white text-center inline-block">
            AICode
          </button>
        </div>
        <div className="flex flex-row items-center justify-center">
          <button 
          onClick={()=>{navigate('/ai-read')}}
          className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-xl leading-[20px] font-medium font-roboto text-white text-left inline-block">
            AIRead
          </button>
        </div>
        <div className="flex flex-row items-center justify-center">
          <button 
          onClick={()=>{navigate('/signup')}}
          className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-xl leading-[20px] font-medium font-roboto text-white text-left inline-block">
            Sign Up
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
